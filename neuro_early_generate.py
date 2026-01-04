import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['VLLM_USE_V1'] = '0'

import json
import argparse
import torch
import numpy as np
import re


from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from jinja2 import Template
from collections import defaultdict
from functools import partial


from vllm.sampling_params import LogitsProcessor




STOP_NEURONS_CONFIG = [


    (22, 3957,   1, 0.46, 27.21),



    (23, 8082,  -1, 0.73, 60.08),

    (20, 16241, -1, 0.12, 22.18),

    (21, 16687,  1, 0.10, 25.63),


    (23, 16705, -1, 0.10, 11.92),
    (22, 16226,  1, 0.09, 11.44),
    (21, 15043, -1, 0.09, 17.78),


    (23, 11748,  1, 0.15, 18.10),
    (20, 12725, -1, 0.06, 15.23),
    (20, 11355, -1, 0.06, 10.76),
]




def set_seeds(seed=42):

    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    g = torch.Generator()
    g.manual_seed(seed)





class EnsembleNeuronMonitor:
    def __init__(self, neuron_config, sim_threshold=0.5, mag_threshold=0.5, prefill_sim_threshold=0.9, calibration_ratio=1.0):
        """
        :param prefill_sim_threshold: Prefill 。
                                       Prefill  (Sim > 0.9)，。
        """
        self.triggered = False
        self.num_neurons = len(neuron_config)
        self.device = 'cpu'
        

        self.sim_threshold = sim_threshold
        self.base_mag_threshold = mag_threshold
        self.dynamic_mag_threshold = mag_threshold
        

        self.prefill_sim_threshold = prefill_sim_threshold
        
        self.calibration_ratio = calibration_ratio
        

        target_acts = [x[4] for x in neuron_config]
        signs_list = [x[2] for x in neuron_config]
        
        self.scale_factors = torch.tensor([1.0 / (t + 1e-6) for t in target_acts], dtype=torch.float32)
        self.ref_signs = torch.tensor(signs_list, dtype=torch.float32)
        self.ref_norm_val = torch.linalg.norm(self.ref_signs) + 1e-10


        self.layer_map = defaultdict(lambda: {'indices': None, 'flat_indices': None})
        self.target_layers = set()
        
        temp_layer_map = defaultdict(lambda: {'indices': [], 'flat_indices': []})
        for idx, (layer, neuron, sign, weight, avg) in enumerate(neuron_config):
            temp_layer_map[layer]['indices'].append(neuron)
            temp_layer_map[layer]['flat_indices'].append(idx)
            self.target_layers.add(layer)
        
        self.max_target_layer = max(self.target_layers)
        
        for layer, data in temp_layer_map.items():
            self.layer_map[layer]['indices'] = torch.tensor(data['indices'], dtype=torch.long)
            self.layer_map[layer]['flat_indices'] = torch.tensor(data['flat_indices'], dtype=torch.long)

        self.current_buffer = None 
        self.epsilon = None 
        self.last_sim = 0.0
        self.last_mag = 0.0

    def to_device(self, device):
        self.device = device
        self.scale_factors = self.scale_factors.to(device)
        self.ref_signs = self.ref_signs.to(device)
        self.ref_norm_val = self.ref_norm_val.to(device)
        self.current_buffer = torch.zeros(self.num_neurons, dtype=torch.float32, device=device)
        self.epsilon = torch.tensor(1e-6, device=device)
        
        for layer in self.target_layers:
            self.layer_map[layer]['indices'] = self.layer_map[layer]['indices'].to(device)
            self.layer_map[layer]['flat_indices'] = self.layer_map[layer]['flat_indices'].to(device)
            
        print(f"✅ Monitor initialized on {device} (Prefill Sim Thr: {self.prefill_sim_threshold})")

    def reset(self):
        self.triggered = False
        self.dynamic_mag_threshold = self.base_mag_threshold
        if self.current_buffer is not None:
            self.current_buffer.fill_(0)
        

        self.last_sim = 0.0
        self.last_mag = 0.0

    def calculate_metrics(self):
        normalized = self.current_buffer * self.scale_factors
        aligned = normalized * self.ref_signs
        current_norm = torch.linalg.norm(aligned)
        
        dot_product = torch.sum(aligned)
        safe_norm = torch.max(current_norm, self.epsilon)
        
        cosine_sim = dot_product / (safe_norm * self.ref_norm_val)
        
        return cosine_sim, current_norm

    def check_prefill_and_calibrate(self):
        """
        【Prefill 】
        """
        sim_tensor, norm_tensor = self.calculate_metrics()
        sim_val = sim_tensor.item()
        norm_val = norm_tensor.item()
        


        if sim_val > self.prefill_sim_threshold and norm_val > 0.2:
            print(f"[Prefill Trigger] Sim: {sim_val:.2f} > {self.prefill_sim_threshold} | Norm: {norm_val:.2f} (STOP)")
            return True


        new_mag_thr = max(self.base_mag_threshold, norm_val * self.calibration_ratio)
        self.dynamic_mag_threshold = new_mag_thr
        


        
        print(f"   [Calib] Prefill(Sim={sim_val:.2f}, Norm={norm_val:.2f}) -> Set Decode Mag Thr({self.dynamic_mag_threshold:.2f})")
        return False

    def check_trigger(self):

        sim_tensor, norm_tensor = self.calculate_metrics()
        

        self.last_sim = sim_tensor.item()
        self.last_mag = norm_tensor.item()
        

        if self.last_sim > self.sim_threshold and self.last_mag > self.dynamic_mag_threshold:
            print(f"[Trigger] Sim: {self.last_sim:.2f} | Norm: {self.last_mag:.2f} (STOP)")
            return True
        return False


    def get_current_metrics(self):
        return self.last_sim, self.last_mag





monitor = EnsembleNeuronMonitor(
    STOP_NEURONS_CONFIG, 
    sim_threshold=0.6, 
    mag_threshold=0.2,
    prefill_sim_threshold=0.8,
    calibration_ratio=0.5
)





def ensemble_mlp_hook(layer_idx, module, input, output):
    with torch.no_grad():
        target_mapping = monitor.layer_map.get(layer_idx)
        if target_mapping is None: return

        num_tokens = output.shape[0]
        

        last_token_vec = output[-1]
        
        selected_vals = torch.index_select(last_token_vec, 0, target_mapping['indices'])
        monitor.current_buffer.index_copy_(0, target_mapping['flat_indices'], selected_vals.float())
        
        if layer_idx == monitor.max_target_layer:
            if num_tokens > 1:


                if monitor.check_prefill_and_calibrate():
                    monitor.triggered = True
            else:

                if monitor.check_trigger():
                    monitor.triggered = True

class Supress_NeuronLogitsProcessor:
    def __init__(self, monitor_ref, stop_token_id, tokenizer):
        self.monitor = monitor_ref
        self.stop_token_id = stop_token_id
        self.tokenizer = tokenizer
        



        reflection_words = [
            "Wait", "But", "Alternatively", "However", "Hmm"
        ]
        self.banned_ids = []
        for word in reflection_words:

            self.banned_ids.extend(tokenizer.encode(word, add_special_tokens=False))
            self.banned_ids.extend(tokenizer.encode(" " + word, add_special_tokens=False))
        self.banned_ids = list(set(self.banned_ids))

    def __call__(self, input_ids, scores):



        sim, mag = self.monitor.get_current_metrics()
        




        if self.monitor.triggered: 

            scores.fill_(-float("inf"))
            scores[self.stop_token_id] = 10000.0
            return scores






        elif sim > 0.4 and mag > 0.2: 

            if("\n" not in self.tokenizer.decode(input_ids[-1])):
                pass
            else:
                scores[self.banned_ids] = -float("inf")

            print(f"Suppressed! Sim={sim:.2f}, Mag={mag:.2f}")




        else:
            pass
            
        return scores


class NeuronStopLogitsProcessor:
    def __init__(self, monitor_ref, stop_token_id, tokenizer):
        self.monitor = monitor_ref
        self.stop_token_id = stop_token_id
        self.neg_inf = float("-inf")
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        if self.monitor.triggered:
            scores.fill_(self.neg_inf)
            scores[self.stop_token_id] = 10000.0
            


            return scores
        return scores


class AntiFillerLogitsProcessor:
    def __init__(self, tokenizer, stop_token_id):
        self.stop_token_id = stop_token_id
        

        fillers = ["Okay", "Alright", "Hmm", "Wait", "OK", "So","Here", "Now","Let","But"]
        
        self.banned_ids = []
        for word in fillers:

            self.banned_ids.extend(tokenizer.encode(word, add_special_tokens=False))
            self.banned_ids.extend(tokenizer.encode(" " + word, add_special_tokens=False))
        

        self.banned_ids = list(set(self.banned_ids))
        self.tokenizer=tokenizer

    def __call__(self, input_ids, scores):

        


        if len(input_ids) == 1:

            scores[self.banned_ids] = -float("inf")
            
        return scores


def register_vllm_hooks(llm_engine):
    try:
        if hasattr(llm_engine.llm_engine.model_executor, 'driver_worker'):
            model = llm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        else:
            model = llm_engine.llm_engine.model_executor.model_runner.model
        device = next(model.parameters()).device
    except Exception as e:
        print(f"Error:  Hook. {e}")
        return False
    
    monitor.to_device(device)
    
    layers = model.model.layers if hasattr(model.model, 'layers') else model.layers
    
    for layer_idx in monitor.target_layers:
        hook_fn = partial(ensemble_mlp_hook, layer_idx)
        layers[layer_idx].mlp.act_fn.register_forward_hook(hook_fn)
    
    print(f"✅ Hook Registered on {len(monitor.target_layers)} layers (Pure GPU).")
    return True






from math_verify import parse, verify
from mathruler.grader import extract_boxed_content, grade_answer







def grade_math_answer(llm_final_answer, gt_answer):
    llm_final_answer = parse(f"${llm_final_answer}$")
    if llm_final_answer is None:
        return 0.0
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = verify(llm_final_answer, parse(f"${gt_answer}$"))
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= verify(llm_final_answer, parse(f"${gt}$"))
    if is_correct:
        return 1.0
    else:
        return 0.0

def grade_gpqa_answer(llm_final_answer, gt_answer):
    if llm_final_answer in gt_answer:
        return 1.0
    else:
        return 0.0

def extract_choice_once_fail(text):
    match = re.findall(r"(?:correct answer is|Answer[:：]?)\s*(?:\*\*)?[\(\[]?([A-E])[\)\]\.\s]?", text, re.IGNORECASE)
    if match: return match[-1].upper()
    match2 = re.findall(r"\b([A-E])\b", text)
    if match2: return match2[-1].upper()
    return "None"

def extract_boxfailed_text(text):

    pattern = r"""
        (?:
            \*\*\s*Answer\s*\*\*
            | \*\*\s*Final\s+Answer\s*\*\*
            | Answer
            | Final\s+Answer
        )
        [:：]?\s*
        (.+)
    """
    
    match = re.findall(pattern, text, re.IGNORECASE | re.VERBOSE | re.DOTALL)
    if match:

        return match[0].strip()
    
    return "None"


def read_jsonl(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="path_to_data")
    parser.add_argument("--data_path", type=str, default="path_to_data")
    parser.add_argument("--output_file", type=str, default="path_to_data")
    args = parser.parse_args()


    print("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=16384,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_prefix_caching=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    

    if not register_vllm_hooks(llm): exit(1)
    


    eos_token_id = tokenizer.eos_token_id


    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples.")



    

    dataset = dataset

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    results_file = open(args.output_file, 'w+', encoding='utf-8')

    for index, datapoint in enumerate(tqdm(dataset)):
        problem = datapoint["problem"]
        template_jinja = """\
        Please reason step by step, and put your final answer within \\boxed{}
        This is the problem:
        {{prompt}}
        """


        prompt_temp=problem+"\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompt = f"<｜User｜>{prompt_temp}<｜Assistant｜><think>\n"




        target_token_id = tokenizer("</think>").input_ids[1]


        monitor.reset()
        






        


        stop_processor = Supress_NeuronLogitsProcessor(
            monitor_ref=monitor, 
            stop_token_id=target_token_id,
            tokenizer=tokenizer,

        
        )







        

        
        anti_filler_processor = AntiFillerLogitsProcessor(tokenizer, target_token_id)


        params_think = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=12000,
            stop=["</think>"],  # 
            n=1,
            seed=42+index,
            logits_processors=[stop_processor]
        )

        neuron_exit_triggered = False
        

        outputs = llm.generate([prompt], params_think, use_tqdm=False)
        
        output_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason
        
        generated_thinking = output_text
        

        print("")
        if monitor.triggered:


            neuron_exit_triggered = True
            if "</think>" not in generated_thinking:
                generated_thinking += "</think>"
        elif finish_reason == "stop":

            print("")
            if "</think>" not in generated_thinking:
                generated_thinking += "</think>"
        elif finish_reason == "length":

            generated_thinking += "</think>"




        current_text = prompt + generated_thinking
        
        params_ans = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            stop=[tokenizer.eos_token],
            n=1,
            seed=42+index,
            logits_processors=[anti_filler_processor]
        )
        
        outputs = llm.generate([current_text], params_ans, use_tqdm=False)
        generated_answer = outputs[0].outputs[0].text
        



        full_response = generated_thinking + generated_answer
        
        if 'answer' in datapoint:
            gt_answer = datapoint['answer']
        else:
            gt_answer = extract_boxed_content(datapoint['solution'])

        llm_final_answer = extract_boxed_content(full_response)
        if llm_final_answer == "None":
            llm_final_answer = extract_choice_once_fail(full_response)
            

        think_idx = full_response.find('</think>')
        if think_idx != -1:
            final_reasoning = full_response[:think_idx].replace(prompt, "").replace("<｜begin of sentence｜>", "")
        else:
            final_reasoning = full_response

        is_correct = grade_math_answer(llm_final_answer, gt_answer)
        
        record = datapoint.copy()
        record['llm_reasoning'] = [final_reasoning]
        record['llm_reasoning_token_num'] = [len(tokenizer.encode(final_reasoning))]
        record['llm_answer'] = [generated_answer]
        record['llm_answer_token_num'] = [len(tokenizer.encode(generated_answer))]
        record['llm_final_answer'] = [llm_final_answer]
        record['is_correct'] = [is_correct]
        record['neuron_triggered'] = neuron_exit_triggered 
        
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_file.flush()

    results_file.close()
    print(f"Finished. Results saved to {args.output_file}")

if __name__ == "__main__":
    set_seeds(seed=42)
    main()
