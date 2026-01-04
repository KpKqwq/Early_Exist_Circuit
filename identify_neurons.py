
from math import log
from transformers import set_seed
set_seed(42)
from collections import Counter
import os
os.environ['CUFILE_FOUND'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import nnsight
from nnsight import LanguageModel
import re
import argparse
import os
import json
import random
import numpy as np
import torch
from typing import List, Tuple
from modeling_qwen2 import Qwen2ForCausalLM
from einops import rearrange
import json
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import bertviz, uuid
from bertviz.util import format_special_chars, format_attention, num_layers, num_heads
import logging
import seaborn as sns
from collections import defaultdict
original_print = print  
logging.basicConfig(level=logging.INFO)
file_path_global = "my_log_file.log"
with open(file_path_global, "w") as f:
    pass

import builtins
def print_log(*args, file_path=file_path_global, **kwargs):
    """ + """
    msg = " ".join(str(a) for a in args)
    with open(file_path, "a") as f:
        f.write(msg + "\n")
    builtins.print(*args, **kwargs)






def get_pos_vector(vector, pos_embed_var):
    vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
    return vector
def transfer_output(model_output):
    all_pos_layer_input = []
    all_pos_attn_output = []
    all_pos_all_pos_residual_output = []
    all_pos_ffn_output = []
    all_pos_layer_output = []
    all_last_attn_subvalues = []
    all_pos_coefficient_scores = []
    all_attn_scores = []
    for layer_i in range(LAYER_NUM):
        cur_layer_input = model_output[layer_i][0]
        cur_attn_output = model_output[layer_i][1]
        cur_all_pos_residual_output = model_output[layer_i][2]
        cur_ffn_output = model_output[layer_i][3]
        cur_layer_output = model_output[layer_i][4]
        cur_last_attn_subvalues = model_output[layer_i][5]
        cur_coefficient_scores = model_output[layer_i][6]
        cur_attn_weights = model_output[layer_i][7]
        all_pos_layer_input.append(cur_layer_input[0].tolist())
        all_pos_attn_output.append(cur_attn_output[0].tolist())
        all_pos_all_pos_residual_output.append(cur_all_pos_residual_output[0].tolist())
        all_pos_ffn_output.append(cur_ffn_output[0].tolist())
        all_pos_layer_output.append(cur_layer_output[0].tolist())
        all_last_attn_subvalues.append(cur_last_attn_subvalues[0].tolist())
        all_pos_coefficient_scores.append(cur_coefficient_scores[0].tolist())
        all_attn_scores.append(cur_attn_weights)
    return all_pos_layer_input, all_pos_attn_output, all_pos_all_pos_residual_output, all_pos_ffn_output, \
           all_pos_layer_output, all_last_attn_subvalues, all_pos_coefficient_scores, all_attn_scores
def get_fc2_params(model, layer_num):
    return model.model.layers[layer_num].mlp.down_proj.weight.detach()

def get_bsvalues_prev(vector, model, final_var, max_tokens: int = 100000):
    """
     get_bsvalues： 4D 
    vector: [*, hidden_dim]
     4D ， flatten  lm_head。
    """







    vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
    vector_rmsn = vector * model_prev.model.norm.weight.detach()
    hidden_dim = vector_rmsn.shape[-1]


    if vector_rmsn.ndim == 4:

        
        vector_flat = vector_rmsn.reshape(-1, hidden_dim)
        

        outputs = []
        with torch.no_grad():
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]

                out = model.lm_head(batch).detach()
                outputs.append(out.cpu())


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    else:



        vector_flat = vector_rmsn.reshape(-1, hidden_dim)


        outputs = []
        lm_head = model.lm_head
        with torch.no_grad():
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]
                out = lm_head(batch).detach()
                outputs.append(out.cpu())


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    return vector_bsvalues




def get_bsvalues_cpu(vector, model, final_var, max_tokens: int = 1000):
    """
     get_bsvalues： 4D 
    vector: [*, hidden_dim]
     4D ， flatten  lm_head。
    """
    vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
    vector_rmsn = vector.cpu() * model.model.norm.weight.detach().cpu()
    hidden_dim = vector_rmsn.shape[-1]

    if vector_rmsn.ndim == 4:

        
        vector_flat = vector_rmsn.reshape(-1, hidden_dim)
        

        outputs = []
        with torch.no_grad():
            lm_head = model.lm_head.cuda()
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]

                out = lm_head(batch.cuda()).detach()
                outputs.append(out.cpu())


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    else:

        vector_flat = vector_rmsn.reshape(-1, hidden_dim)


        outputs = []
        with torch.no_grad():
            lm_head = model.lm_head.cuda()
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]
                out = lm_head(batch.cuda()).detach()
                outputs.append(out)


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    return vector_bsvalues



def get_bsvalues(vector, model, final_var, max_tokens: int = 1000):
    """
     get_bsvalues： 4D 
    vector: [*, hidden_dim]
     4D ， flatten  lm_head。
    """







    vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
    vector_rmsn = vector * model.model.norm.weight.detach()
    hidden_dim = vector_rmsn.shape[-1]


    if vector_rmsn.ndim == 4:

        
        vector_flat = vector_rmsn.reshape(-1, hidden_dim)
        

        outputs = []
        with torch.no_grad():
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]

                out = model.lm_head(batch).detach()
                outputs.append(out.cpu())


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    else:



        vector_flat = vector_rmsn.reshape(-1, hidden_dim)


        outputs = []
        lm_head = model.lm_head
        with torch.no_grad():
            for i in range(0, vector_flat.size(0), max_tokens):
                batch = vector_flat[i:i + max_tokens]
                out = lm_head(batch).detach()
                outputs.append(out.cpu())


        vector_bsvalues = torch.cat(outputs, dim=0)
        new_shape = list(vector_rmsn.shape[:-1]) + [vector_bsvalues.shape[-1]]
        vector_bsvalues = vector_bsvalues.view(*new_shape)
    return vector_bsvalues

def get_bsvalues_fast(vector, model, final_var):
    with torch.no_grad():
        vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
        vector_rmsn = vector * model.model.norm.weight.detach()

        vector_bsvalues = model.lm_head(vector_rmsn).detach()
    return vector_bsvalues  

def get_bsvalues_fast_nonorm(vector, model):
    with torch.no_grad():



        vector_bsvalues = model.lm_head(vector).detach()
    return vector_bsvalues  


def get_layernorm_weight(model, layer_num):
    return model.model.layers[layer_num].post_attention_layernorm.weight.detach()
def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob

def get_log_increase(cur_layer_input, cur_attn_vector_list, final_var, predict_index):
    with torch.no_grad():
        origin_prob_log = torch.log(get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))[predict_index])

        cur_attn_vector_plus = cur_layer_input.detach().clone()
        for vec in cur_attn_vector_list:
            cur_attn_vector_plus += vec
        cur_attn_vector_bsvalues = get_bsvalues(cur_attn_vector_plus, model, final_var)
        cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)
        cur_attn_vector_probs = cur_attn_vector_probs[predict_index]
        cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)
        cur_attn_vector_probs_log_increase = cur_attn_vector_probs_log - origin_prob_log

    return cur_attn_vector_probs_log_increase


def get_log_increase_with_probs(cur_layer_input, cur_attn_vector_list, final_var, predict_index):
    with torch.no_grad():

        prob_log = get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))
        origin_prob_log = torch.log(get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))[predict_index])
        rank = (torch.argsort(prob_log, descending=True) == predict_index).nonzero(as_tuple=True)[0].item()

        cur_attn_vector_plus = cur_layer_input.detach().clone()
        for vec in cur_attn_vector_list:
            cur_attn_vector_plus += vec
        cur_attn_vector_bsvalues = get_bsvalues(cur_attn_vector_plus, model, final_var)
        cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)
        cur_attn_vector_probs = cur_attn_vector_probs[predict_index]
        cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)
        cur_attn_vector_probs_log_increase = cur_attn_vector_probs_log - origin_prob_log

    return cur_attn_vector_probs_log_increase,torch.exp(origin_prob_log),rank

def get_log_increase_with_probs_decrease(cur_layer_input, cur_attn_vector_list, final_var, predict_index):
    with torch.no_grad():

        prob_log = get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))
        origin_prob_log = torch.log(get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))[predict_index])


        cur_attn_vector_plus = cur_layer_input.detach().clone()
        for vec in cur_attn_vector_list:
            cur_attn_vector_plus -= vec
        cur_attn_vector_bsvalues = get_bsvalues(cur_attn_vector_plus, model, final_var)
        cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)
        rank = (torch.argsort(cur_attn_vector_probs, descending=True) == predict_index).nonzero(as_tuple=True)[0].item()
        cur_attn_vector_probs = cur_attn_vector_probs[predict_index]
        cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)
        cur_attn_vector_probs_log_increase = origin_prob_log-cur_attn_vector_probs_log 

    return cur_attn_vector_probs_log_increase,torch.exp(cur_attn_vector_probs_log),rank



def get_log_increase_with_probs_decrease_neuron(cur_layer_input, cur_attn_vector_list, final_var, predict_index):
    with torch.no_grad():

        prob_log = get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))
        origin_prob_log = torch.log(get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))[:,predict_index])


        cur_attn_vector_plus = cur_layer_input.detach().clone()
        for vec in cur_attn_vector_list:

            cur_attn_vector_plus = cur_attn_vector_plus-vec
        cur_attn_vector_bsvalues = get_bsvalues(cur_attn_vector_plus, model, final_var)
        cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)

        rank = (torch.argsort(cur_attn_vector_probs, descending=True) == predict_index).nonzero(as_tuple=True)[1]
        cur_attn_vector_probs = cur_attn_vector_probs[:,predict_index]
        cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)

        cur_attn_vector_probs_log_increase = origin_prob_log-cur_attn_vector_probs_log.cuda() 

    return cur_attn_vector_probs_log_increase,torch.exp(cur_attn_vector_probs_log),rank


def get_log_increase_P_vs_N(cur_layer_input, layer_i,layer_type, final_var, predict_index):
    coefficient_scores = torch.tensor(all_pos_coefficient_scores[layer_i][-1])
    fc2_vectors = get_fc2_params(model, layer_i)
    ffn_subvalues = (coefficient_scores * fc2_vectors).T
    origin_prob_log = torch.log(get_prob(get_bsvalues(cur_layer_input, model, final_var))[predict_index])
    cur_attn_vector_plus = cur_layer_input+ffn_subvalues
    if(layer_i==25):
        import pdb;pdb.set_trace()
    cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(cur_attn_vector_plus, model, final_var))[:, predict_index])
    cur_attn_vector_probs_log_increase = cur_attn_plus_probs - origin_prob_log
    return cur_attn_vector_probs_log_increase


def plt_bar(x, y, yname="log increase",name="Wait"):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    plt.ylim(-2, 10)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.show()
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png")

def plt_bar_headmap(x, y, yname="log increase", name="Wait"):


    num_layers = len(x) // 2 + len(x) % 2
    data = np.zeros((2, num_layers))

    attn_idx = 0
    ffn_idx = 0
    for i in range(len(x)):
        if i % 2 == 0:
            data[0, attn_idx] = y[i]
            attn_idx += 1
        else:
            data[1, ffn_idx] = y[i]
            ffn_idx += 1

    plt.figure(figsize=(10, 2))
    sns.heatmap(data, annot=False, cmap="YlGnBu", xticklabels=[f"L{i}" for i in range(num_layers)],
                yticklabels=["Attention", "FFN"],vmin=-1,vmax=5)
    plt.xlabel("Layer")
    plt.ylabel("Type")
    plt.title(yname)
    

    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png", bbox_inches='tight')
    plt.show()

def plot_bar_heatmap_ffn(data, yname="Value", name="Layer Heatmap"):
    """
    data: list of [name, value], e.g. [['27_6621', 0.46875], ['25_596', 0.375], ...]
    """

    layer_dict = defaultdict(float)
    max_layer = 0
    for item in data:
        layer_str, val = item
        layer_num = int(layer_str.split('_')[0])
        layer_dict[layer_num] += val
        if layer_num > max_layer:
            max_layer = layer_num


    heatmap_values = [layer_dict[i] if i in layer_dict else 0 for i in range(max_layer + 1)]
    

    heatmap_data = np.array([heatmap_values])
    

    plt.figure(figsize=(max_layer * 0.4 + 2, 2))
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap="YlGnBu",
        xticklabels=[f"L{i}" for i in range(max_layer + 1)],
        yticklabels=False,
        cbar_kws={'label': yname}
    )
    plt.xlabel("Layer")
    plt.title(name)
    

    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png", bbox_inches='tight')
    plt.show()



def plot_bar_heatmap_attention(data, yname="Value", name="Attention Heatmap"):
    """
    data: list of (id_string, value)
    id_string example: '27_10_100_22'
    first number: layer index
    last number: position index
    ，（layer=0 ）
    """

    value_dict = defaultdict(float)
    max_layer, max_pos = 0, 0

    for id_str, val in data:
        parts = id_str.split('_')
        layer_idx = int(parts[0])
        pos_idx = int(parts[-1])
        value_dict[(layer_idx, pos_idx)] += val
        max_layer = max(max_layer, layer_idx)
        max_pos = max(max_pos, pos_idx)


    heatmap_data = np.zeros((max_layer + 1, max_pos + 1))
    for (layer, pos), val in value_dict.items():
        heatmap_data[layer, pos] = val


    vmin, vmax = np.min(heatmap_data), np.max(heatmap_data)
    if vmin >= 0:
        vmin = -vmax * 0.05
    elif vmax <= 0:
        vmax = -vmin * 0.05


    colors = [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


    plt.figure(figsize=(max_pos * 0.4 + 2, max_layer * 0.3 + 2))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        norm=norm,
        cbar=True,
        xticklabels=[f"P{i}" for i in range(max_pos + 1)],
        yticklabels=[f"L{i}" for i in range(max_layer + 1)],
        cbar_kws={'label': yname}
    )


    plt.gca().invert_yaxis()

    plt.xlabel("Position index")
    plt.ylabel("Layer index")
    plt.title(name)


    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.tight_layout()
    plt.savefig(f"attention_neurons.{safe_name}.png", bbox_inches='tight', dpi=300)
    plt.show()

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

def plot_bar_heatmap_query_layer_position(curfile_ffn_score_dict, name="query_ffn_heatmap"):
    """
     query FFN neuron -
    :
        curfile_ffn_score_dict: dict, key  "layer_neuron_position"
        name: 
    """

    layer_pos_scores = defaultdict(float)
    max_layer, max_pos = 0, 0

    for key, value in curfile_ffn_score_dict.items():
        parts = key.split("_")
        if len(parts) < 3:
            continue
        layer = int(parts[0])
        pos = int(parts[-1])
        layer_pos_scores[(layer, pos)] += float(value)
        max_layer = max(max_layer, layer)
        max_pos = max(max_pos, pos)


    heatmap = np.zeros((max_layer + 1, max_pos + 1))
    for (layer, pos), val in layer_pos_scores.items():
        heatmap[layer, pos] = val


    vmin, vmax = np.min(heatmap), np.max(heatmap)
    colors = [
        (0.0, "blue"),
        (0.5, "white"),
        (1.0, "red")
    ]
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap,
        cmap=cmap,
        cbar=True,
        square=False,
        xticklabels=True,
        yticklabels=True,
        norm=norm
    )


    plt.gca().invert_yaxis()

    plt.xlabel("Position index")
    plt.ylabel("Layer index")
    plt.title("Query FFN contribution heatmap (layer 0 at bottom)")


    safe_name = name.replace("/", "_").replace("\\", "_")
    plt.tight_layout()
    plt.savefig(f"{safe_name}.png", dpi=300)
    plt.show()



def plt_bar_ffn_neuron(x, y, yname="log increase",name="Wait"):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.ylim(-0.2, 0.5)

    plt.show()
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png")


def plt_bar_ffn_range(x, y, yname="log increase",name="Wait",first=-1,second=-1):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.ylim(first, second)

    plt.show()
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png")

def plt_heatmap(data):
    data = np.array(data)
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    


    cmap = plt.cm.hot_r
    cmap.set_under('white')
    

    im = ax.imshow(data, cmap=cmap, vmin=0)
    

    cbar = plt.colorbar(im)
    cbar.set_label('Value')
    

    plt.title("attn head log increase heatmap")
    plt.tight_layout()
    

    plt.show()
    

    plt.savefig("attn_head_log_increase_heatmap.png")


def plt_heatmap_range(data):
    data = np.array(data)
    

    data[np.abs(data) < 0.01] = 0
    
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)
    
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    

    cmap = plt.cm.seismic  
    

    vmax = np.max(np.abs(data))
    vmin = -vmax
    

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    

    cbar = plt.colorbar(im)
    cbar.set_label('Value (|x| < 0.01 → 0)')
    
    plt.title("attn head log increase heatmap (|x|<0.01 set to 0)")
    plt.tight_layout()
    
    plt.savefig("attn_head_log_increase_heatmap_range.png", dpi=300)
    plt.show()


def transfer_l(l):
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y
import math
def plot_attention_heatmaps(
    all_attn_scores,
    save_dir="attn_heatmaps",
    fold=50,
    source_tokens=None,
    target_tokens=None,
    vmax=None,
    vmin=None
):
    """
     attention ， fold  target token 。
    
    :
        all_attn_scores: [num_layers, num_heads, source_len, target_len]  ( torch.Tensor)
        save_dir: 
        fold:  target token 
        source_tokens: list[str]  None， y 
        target_tokens: list[str]  None， x 
        vmax, vmin: ，
    """
    os.makedirs(save_dir, exist_ok=True)
    num_layers = len(all_attn_scores)
    num_heads = len(all_attn_scores[0])
    cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = all_attn_scores[layer_idx][head_idx]


            if hasattr(attn, "detach"):
                attn = attn.detach().to(dtype=torch.float32).cpu().numpy()
            else:
                attn = np.array(attn, dtype=np.float32)

            source_len, target_len = attn.shape
            num_rows = math.ceil(target_len / fold)

            fig, axes = plt.subplots(num_rows, 1, figsize=(12, 2.5 * num_rows), squeeze=False)
            axes = axes.flatten()

            for row in range(num_rows):
                t_start = row * fold
                t_end = min((row + 1) * fold, target_len)
                ax = axes[row]


                chunk = attn[:, t_start:t_end]
                img = ax.imshow(chunk, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)


                if target_tokens is not None:
                    ax.set_xticks(np.arange(t_end - t_start))
                    ax.set_xticklabels(target_tokens[t_start:t_end], rotation=90, fontsize=8)
                else:
                    ax.set_xlabel(f"Target {t_start}–{t_end-1}")


                if source_tokens is not None:
                    ystep = max(1, source_len // 20)
                    yticks = np.arange(0, source_len, ystep)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([source_tokens[i] for i in yticks], fontsize=8)
                else:
                    ax.set_ylabel("Source")

                ax.set_title(f"Targets {t_start}–{t_end-1}")

            fig.suptitle(f"Layer {layer_idx} | Head {head_idx}", fontsize=14)
            fig.colorbar(img, ax=axes.tolist(), orientation='vertical', fraction=0.02, pad=0.02)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fname = os.path.join(save_dir, f"layer{layer_idx}_head{head_idx}.png")
            plt.show()
            plt.savefig(fname, dpi=150)
            plt.close(fig)

    print(f"✅  {save_dir}")




from transformers import AutoTokenizer










parser = argparse.ArgumentParser(description="Inference script for language models.")
parser.add_argument("--model_path", type=str, default="path_to_data", help="Path to the model")
parser.add_argument("--output_data", action="store_true", help="Whether to output data_to_save")

args, unknown = parser.parse_known_args()
data_path="path_to_data"





model = LanguageModel(args.model_path, device_map='cuda', torch_dtype=torch.bfloat16,automodel=Qwen2ForCausalLM,attn_implementation="eager")






LAYER_NUM = 28
HEAD_NUM = 128
HEAD_DIM = 12
HIDDEN_DIM = HEAD_NUM * HEAD_DIM
torch.set_default_device("cuda")
all_selected_neurons=[]
data_path="path_to_data"


count1=0
for line in open(data_path, 'r', encoding='utf-8'):
    count1+=1


    print("Finish one line")
    tmp_data = json.loads(line)
    problem = tmp_data["problem"]
    reason_str = tmp_data["llm_reasoning"]




    input_text =f"<｜begin▁of▁sentence｜><｜User｜>{problem}<｜Assistant｜><think>\n{reason_str[0]}"

    input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids[:,1:]

    attention_mask = torch.ones_like(input_ids)
    all_pos_residual_output=[]


    indexed_tokens = model.tokenizer.encode(input_text,add_special_tokens=False)

    tokens = [model.tokenizer.decode(x) for x in indexed_tokens]



    answer="</think>"
    no_words=["\n","\n\n"]



    predict_index_list=[]
    wait_positions = []
    for idx, tok in enumerate(tokens):
        for word1 in no_words:
            if(word1 in tok):
                wait_positions.append(idx)
                predict_index_list.append(model.tokenizer.encode(answer)[1])
                break
    print_log("Wait token positions:", wait_positions)
    for wait_index in range(len(wait_positions)):
        if answer in "".join(tokens[:wait_positions[wait_index]]):
            print("Finding first answer pos",wait_positions[wait_index])
            break

    INPUT_LEN=len(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        with model.trace({"input_ids": input_ids, "attention_mask": attention_mask},output_attentions=True) as tracer:
            hidden_states = []
            attention_states=[]
            attentions = []
            ffn_pre_act = []
            ffn_swiglu_out = []
            all_last_attn_subvalues = []
            layer_output_list = []
            all_pos_layer_input = []
            all_pos_layer_output=[]
            all_pos_attn_output=[]
            all_attn_scores=[]
            all_pos_ffn_output=[]
            all_pos_coefficient_scores=nnsight.list().save()
            all_pos_gate_scores=nnsight.list().save()
            for layer in model.model.layers:
                all_pos_layer_input.append(layer.input[0].cpu().save())
                all_pos_attn_output.append(layer.self_attn.output[0].cpu().save())
                all_attn_scores.append(layer.self_attn.output[1].cpu().save())

                all_pos_ffn_output.append(layer.mlp.output[0].cpu().save())
                all_pos_layer_output.append(layer.output[0].cpu().save())
                all_last_attn_subvalues.append(layer.self_attn.output[-1].cpu().save())
                all_pos_coefficient_scores.append((layer.mlp.act_fn(layer.mlp.gate_proj(layer.mlp.input[0])) * layer.mlp.up_proj(layer.mlp.input[0])).cpu())


    for idx, item in enumerate(all_pos_attn_output):


        new_item = item + all_pos_layer_input[idx]
        all_pos_residual_output.append(new_item.detach())



    for index in range(LAYER_NUM):

        all_last_attn_subvalues[index]=all_last_attn_subvalues[index][0].to(torch.bfloat16).tolist()
        all_pos_layer_output[index]=all_pos_layer_output[index][0].to(torch.bfloat16).tolist()
        
        all_pos_layer_input[index]=all_pos_layer_input[index].to(torch.bfloat16).tolist()
        all_pos_attn_output[index]=all_pos_attn_output[index][0].to(torch.bfloat16).tolist()

        all_pos_ffn_output[index]=all_pos_ffn_output[index].to(torch.bfloat16).tolist()
        all_pos_residual_output[index]=all_pos_residual_output[index][0].to(torch.bfloat16).tolist()


    final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)



    wait_words = ["</think>"]
    no_wait_words = ["\n"]

    predict_index_tmp=model.tokenizer.encode("</think>")[1]
    no_predict_index=model.tokenizer.encode("\n")[1]


    def find_import_layers(pos=-1,predict_index=predict_index_tmp,final_var=final_var):
        all_attn_log_increase = []
        origin_prob_log = get_prob(get_bsvalues_fast(torch.tensor(all_pos_layer_output[-1][pos]), model, final_var))[predict_index]
        sorted_probs, sorted_indices = torch.sort(get_prob(get_bsvalues_fast(torch.tensor(all_pos_layer_output[-1][pos]), model, final_var)), descending=True)
        
        print("Original probability:",origin_prob_log)
        print("Sorted tokens:", [model.tokenizer.decode(int(x)) for x in sorted_indices[:10]])
        print("Sorted prob:", sorted_probs[:10].tolist())
        
        for layer_i in range(LAYER_NUM):
            cur_attn_vector = torch.tensor(all_pos_attn_output[layer_i][pos])
            cur_layer_input = torch.tensor(all_pos_layer_input[layer_i][pos])
            cur_attn_vector_probs_log_increase = get_log_increase(cur_layer_input, [cur_attn_vector], final_var, predict_index)
            all_attn_log_increase.append(cur_attn_vector_probs_log_increase.item())
        all_ffn_log_increase = []
        for layer_i in range(LAYER_NUM):
            cur_ffn_vector = torch.tensor(all_pos_ffn_output[layer_i][pos])
            cur_residual = torch.tensor(all_pos_residual_output[layer_i][pos])
            cur_residual = cur_residual.to(torch.bfloat16)
            cur_ffn_vector_probs_log_increase = get_log_increase(cur_residual, [cur_ffn_vector], final_var, predict_index)
            all_ffn_log_increase.append(cur_ffn_vector_probs_log_increase.tolist())
        attn_list, ffn_list = [], []
        for layer_i in range(LAYER_NUM):
            attn_list.append([str(layer_i), all_attn_log_increase[layer_i]])
            ffn_list.append([str(layer_i), all_ffn_log_increase[layer_i]])
        attn_list_sort = attn_list[15:]
        ffn_list_sort = ffn_list[15:]
        attn_increase_compute, ffn_increase_compute = [], []
        for indx, increase in attn_list_sort:
            attn_increase_compute.append((indx, round(increase, 3)))
        for indx, increase in ffn_list_sort:
            ffn_increase_compute.append((indx, round(increase, 3)))
        print_log("attn sum: ", sum([x[1] for x in attn_increase_compute]), 
            "ffn sum: ", sum([x[1] for x in ffn_increase_compute]))
        print_log("attn: ", attn_increase_compute)
        print_log("\n")
        print_log("ffn: ", ffn_increase_compute)
        all_increases_draw = []
        for i in range(len(attn_list)):
            all_increases_draw.append(attn_list[i][1])
            all_increases_draw.append(ffn_list[i][1])    
        plt_bar(range(len(all_increases_draw)), all_increases_draw,name=wait_words[0])
        plt_bar_headmap(range(len(all_increases_draw)), all_increases_draw,name=wait_words[0])
        return attn_increase_compute, ffn_increase_compute


    from math import exp
    def find_ffn_value_neurons(pos=-1,predict_index=predict_index_tmp,final_var=final_var):
        all_ffn_subvalues = []
        for layer_i in range(LAYER_NUM):
            coefficient_scores = torch.tensor(all_pos_coefficient_scores[layer_i][pos])
            fc2_vectors = get_fc2_params(model, layer_i)
            ffn_subvalues = (coefficient_scores * fc2_vectors).T
            all_ffn_subvalues.append(ffn_subvalues)
        ffn_subvalue_list = []
        for layer_i in range(19,LAYER_NUM):
            cur_ffn_subvalues = all_ffn_subvalues[layer_i]
            cur_residual = torch.tensor(all_pos_residual_output[layer_i][pos])

            origin_prob_log = torch.log(get_prob(get_bsvalues(cur_residual, model, final_var))[predict_index])

            cur_ffn_subvalues_plus = cur_ffn_subvalues + cur_residual
            

            cur_ffn_subvalues_bsvalues = get_bsvalues(cur_ffn_subvalues_plus, model, final_var)
            cur_ffn_subvalues_probs = get_prob(cur_ffn_subvalues_bsvalues)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs[:, predict_index]
            cur_ffn_subvalues_probs_log = torch.log(cur_ffn_subvalues_probs)
            cur_ffn_subvalues_probs_log_increase = cur_ffn_subvalues_probs_log - origin_prob_log
            for index, ffn_increase in enumerate(cur_ffn_subvalues_probs_log_increase):
                ffn_subvalue_list.append([str(layer_i)+"_"+str(index), ffn_increase.item()])
        ffn_subvalue_list_sort = sorted(ffn_subvalue_list, key=lambda x: x[-1])[::-1]
        for x in ffn_subvalue_list_sort[:10]:
            print_log(x[0], round(x[1], 4))
            layer = int(x[0].split("_")[0])
            neuron = int(x[0].split("_")[1])
            cur_vector = get_fc2_params(model, layer).T[neuron]
            cur_vector_bsvalue = get_prob(get_bsvalues(cur_vector, model, final_var))
            sorted_values,cur_vector_bsvalue_sort = torch.sort(cur_vector_bsvalue, descending=True)

            predict_rank = (cur_vector_bsvalue_sort == predict_index).nonzero(as_tuple=True)[0].item()
            no_predict_rank = (cur_vector_bsvalue_sort == no_predict_index).nonzero(as_tuple=True)[0].item()







        positive_ffn = [(k, v) for k, v in ffn_subvalue_list_sort if v>0]
        FFN_value_neurons = [x[0] for x in positive_ffn[:300]]
        total_neurons = len(FFN_value_neurons)
        FFN_layer_count_value = [int(x.split("_")[0]) for x in list(FFN_value_neurons)]
        FFN_layer_count_value = Counter(FFN_layer_count_value)
        FFN_layer_count_value = sorted(zip(FFN_layer_count_value.keys(), FFN_layer_count_value.values()))
        gpt_FFN_value_x, gpt_FFN_value_y = transfer_l(FFN_layer_count_value)


        plt.figure(figsize=(6,3))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.plot(gpt_FFN_value_x, gpt_FFN_value_y, "bo-", label="FFN value neurons")
        plt.xlabel("layer", fontsize=10)
        plt.ylabel("count", fontsize=10)
        plt.text(0.05, 0.95, f"Total neurons: {total_neurons}", fontsize=10, transform=plt.gca().transAxes, ha="left", va="top", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
        plt.legend(fontsize=10, loc="upper right")
        plt.show()


        return ffn_subvalue_list_sort[:300]

    all_wait_important_ffn_neurons = []
    for index,pos in enumerate(wait_positions[-1:]):
        print("Draw position:", pos, "index:", index)
        final_var = torch.tensor(all_pos_layer_output[-1][pos]).pow(2).mean(-1, keepdim=True)

        ffn_subvalue_list_sort_forward_pos=find_ffn_value_neurons(pos=pos,predict_index=predict_index_list[index],final_var=final_var)
        all_wait_important_ffn_neurons.append(ffn_subvalue_list_sort_forward_pos)





    from collections import defaultdict
    sum_dict = defaultdict(float)


    for sublist in all_wait_important_ffn_neurons:
        for k, v in sublist:

            if(int(k.split("_")[0])<=10):
                
                continue
            sum_dict[k] += v






    sorted_result = sorted(sum_dict.items(), key=lambda x: x[1], reverse=True)



    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import entropy

    neuron_data = []
    seq_len = len(tokens)
    positions = np.arange(seq_len)


    target_pos_indices = wait_positions[-1:] 

    print(f" []  {len(sorted_result[:100])} ...")

    for neuro_name, importance_score in sorted_result[:100]:
        layer_index, neuro_index = map(int, neuro_name.split("_"))

        raw_acts = np.array([
            all_pos_coefficient_scores[layer_index][pos][neuro_index].item() 
            for pos in range(seq_len)
        ])
        


        final_wait_pos = wait_positions[-1]
        final_val = all_pos_coefficient_scores[layer_index][final_wait_pos][neuro_index].item()
        

        target_sign = -1 if final_val < 0 else 1
        


        aligned_acts = raw_acts * target_sign
        


        rectified_acts = np.maximum(aligned_acts, 0)

        total_act_sum = np.sum(rectified_acts)
        if total_act_sum < 1e-6:
            continue
            

        

        probs = rectified_acts / total_act_sum
        ent = entropy(probs, base=2)
        

        center_of_mass = np.sum(positions * rectified_acts) / total_act_sum
        relative_com = center_of_mass / seq_len
        
        neuron_data.append({
            "name": neuro_name,
            "layer": layer_index,
            "neuron": neuro_index,
            "importance": importance_score,
            "entropy": ent,
            "relative_com": relative_com,
            "target_sign": target_sign,
            "aligned_acts": aligned_acts
        })


    sorted_neurons = sorted(
        neuron_data, 
        key=lambda x: (x["relative_com"], -x["entropy"]), 
        reverse=True
    )

    top_k = 100
    selected_neurons = sorted_neurons[:top_k]
    all_selected_neurons.append(selected_neurons)

import pandas as pd
from collections import defaultdict




total_samples = len(all_selected_neurons)
print(f" {total_samples} ...")




neuron_stats = defaultdict(lambda: {
    "count": 0, 
    "total_score": 0.0, 
    "total_com": 0.0,
    "layer": 0,
    "neuron_idx": 0,
    "activation": 0.0
})




for sample_neurons in all_selected_neurons:


    voting_candidates = sample_neurons[:50]
    
    for item in voting_candidates:
        name = item['name']
        stats = neuron_stats[name]
        
        stats["count"] += 1
        stats["total_score"] += item['importance']
        stats["total_com"] += item['relative_com']
        stats["layer"] = item['layer']
        stats["neuron_idx"] = item['neuron']
        stats["activation"]+=item['aligned_acts'][-1]


data_list = []
for name, stats in neuron_stats.items():
    count = stats["count"]
    data_list.append({
        "Name": name,
        "Layer": stats["layer"],
        "Neuron": stats["neuron_idx"],
        "Frequency": count,
        "Consistency": count / total_samples,
        "Avg_Importance": stats["total_score"] / count,
        "Avg_CoM": stats["total_com"] / count,
        "Avg_act": stats["activation"] / count
    })

df = pd.DataFrame(data_list)




filtered_df = df[
    (df["Consistency"] >= 0.6) &
    (df["Avg_CoM"] >= 0.6)
].sort_values(by=["Frequency", "Avg_Importance"], ascending=False)

print(f"\n {len(filtered_df)} ：")
print("-" * 60)

print(filtered_df.head(20).to_string(index=False))


import matplotlib.pyplot as plt

if not filtered_df.empty:
    plt.figure(figsize=(10, 5))

    plt.scatter(
        filtered_df["Layer"], 
        filtered_df["Frequency"], 
        c=filtered_df["Avg_Importance"], 
        cmap='viridis', 
        s=100, 
        alpha=0.7
    )
    plt.colorbar(label='Avg Importance Score')
    plt.xlabel("Layer Index")
    plt.ylabel("Frequency (Count)")
    plt.title("Distribution of Universal Stop Neurons")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
else:
    print("， Consistency 。")


universal_stop_neurons = filtered_df["Name"].tolist()
print("\n (Python list):")
print(universal_stop_neurons[:20])







import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import seaborn as sns





STOP_NEURONS_CONFIG = [
    (20, 8425, -1, 0.27, 10.33),
    (21, 2745, -1, 0.17, 17.08),
    (23, 3982, -1, 0.51, 43.15),
    (21, 7881,  1, 0.09,  5.28),
    (21, 3568,  1, 0.07,  9.76),
    (23, 2436,  1, 0.08, 15.60),
    (21, 7816,  1, 0.06, 10.92),
    (20, 4194, -1, 0.08,  9.29),
    (22, 1525,  1, 0.07,  7.47),
    (21, 2893, -1, 0.05,  9.32),
    (19, 2925, -1, 0.06,  5.77),
]



ref_vector = np.array([x[4] for x in STOP_NEURONS_CONFIG])
ref_norm = norm(ref_vector)


ref_signs = np.array([x[2] for x in STOP_NEURONS_CONFIG])

print(f" (Avg_act) ，: {ref_vector.shape}")
print(f": {ref_norm:.2f}")





num_neurons = len(STOP_NEURONS_CONFIG)

activation_matrix = np.zeros((seq_len, num_neurons))

print("...")

for col_idx, (layer_idx, neuron_idx, sign, weight, avg_act) in enumerate(STOP_NEURONS_CONFIG):
    try:
        raw_acts_list = []
        for t in range(seq_len):

            val = all_pos_coefficient_scores[layer_idx][t][neuron_idx].item()
            raw_acts_list.append(val)
        
        raw_acts = np.array(raw_acts_list)
        



        aligned_acts = raw_acts * sign
        
        activation_matrix[:, col_idx] = aligned_acts
        
    except Exception as e:
        print(f" {layer_idx}_{neuron_idx} : {e}")





similarities = []
magnitudes = []


eps = 1e-10

for t in range(seq_len):

    current_vec = activation_matrix[t]
    


    rectified_vec = np.maximum(current_vec, 0)
    

    current_norm = norm(rectified_vec)
    



    if current_norm < eps:
        sim = 0.0
    else:
        sim = np.dot(rectified_vec, ref_vector) / (current_norm * ref_norm)
    
    similarities.append(sim)
    magnitudes.append(current_norm)

similarities = np.array(similarities)
magnitudes = np.array(magnitudes)




plt.figure(figsize=(20, 12))


ax1 = plt.subplot(3, 1, 1)
ax1.plot(positions, similarities, color='royalblue', linewidth=2, label="Pattern Similarity (Cosine)")

ax1.axhline(0.85, color='orange', linestyle='--', alpha=0.7, label="Threshold (0.85)")
ax1.set_ylabel("Cosine Similarity", fontsize=12)
ax1.set_title("Metric 1: Pattern Matching (Is the shape correct?)", fontsize=14)
ax1.set_ylim(-0.1, 1.1)
ax1.legend(loc="upper left")


for wp in wait_positions:
    ax1.axvline(wp, color='green', linestyle='-', alpha=0.5, linewidth=2)
    ax1.text(wp, 1.02, "GT", color='green', ha='center', fontsize=8)


ax2 = plt.subplot(3, 1, 2)
ax2.plot(positions, magnitudes, color='crimson', linewidth=2, label="Activity Magnitude")

noise_threshold = 5.0 
ax2.axhline(noise_threshold, color='gray', linestyle='--', label=f"Noise Gate ({noise_threshold})")

ax2.set_ylabel("Magnitude (L2 Norm)", fontsize=12)
ax2.set_title("Metric 2: Signal Strength (Is it just noise?)", fontsize=14)
ax2.legend(loc="upper left")
ax2.set_xlim(0, seq_len)

for wp in wait_positions:
    ax2.axvline(wp, color='green', linestyle='-', alpha=0.3)


ax3 = plt.subplot(3, 1, 3)

sns.heatmap(np.maximum(activation_matrix, 0).T, ax=ax3, cmap="magma", cbar=True, vmin=0)
ax3.set_ylabel("Neurons (Index in Config)", fontsize=12)
ax3.set_xlabel("Token Position", fontsize=12)
ax3.set_title("Visual Verification: Neuron Activations (ReLU & Aligned)", fontsize=12)
ax3.set_xticks(positions[::2])
ax3.set_xticklabels(tokens[::2], rotation=90, fontsize=8)

plt.tight_layout()
plt.show()





SIM_THRESHOLD = 0.6
MAG_THRESHOLD = 10.0

print(f"\n【】 Sim > {SIM_THRESHOLD} AND Mag > {MAG_THRESHOLD}\n")


candidates = []
for idx in range(seq_len):
    sim = similarities[idx]
    mag = magnitudes[idx]
    
    if sim > SIM_THRESHOLD and mag > MAG_THRESHOLD:
        token_str = tokens[idx].replace('\n', '\\n')

        hit_gt = "✅ HIT GT" if any(abs(idx - wp) <= 1 for wp in wait_positions) else ""
        print(f"Step {idx:3d} [{token_str:>10}] | Sim: {sim:.3f} | Mag: {mag:.2f} | {hit_gt}")








