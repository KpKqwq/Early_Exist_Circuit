import os
os.environ['CUFILE_FOUND'] = 'false'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import argparse
import torch
import numpy as np
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple
from functools import partial

import nnsight
from nnsight import LanguageModel
from transformers import AutoTokenizer, set_seed
from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.stats import entropy


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_pos_vector(vector, pos_embed_var, final_var):
    vector = vector.to(torch.bfloat16) * torch.rsqrt(final_var + 1e-6).to(torch.bfloat16)
    return vector


def transfer_output(model_output, LAYER_NUM):
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


def get_log_increase(cur_layer_input, cur_attn_vector_list, final_var, predict_index, model, get_bsvalues_fast, get_prob):
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


def get_log_increase_with_probs(cur_layer_input, cur_attn_vector_list, final_var, predict_index, model, get_bsvalues_fast, get_prob):
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
    return cur_attn_vector_probs_log_increase, torch.exp(origin_prob_log), rank


def get_log_increase_with_probs_decrease(cur_layer_input, cur_attn_vector_list, final_var, predict_index, model, get_bsvalues_fast, get_prob):
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
        cur_attn_vector_probs_log_increase = origin_prob_log - cur_attn_vector_probs_log
    return cur_attn_vector_probs_log_increase, torch.exp(cur_attn_vector_probs_log), rank


def get_log_increase_with_probs_decrease_neuron(cur_layer_input, cur_attn_vector_list, final_var, predict_index, model, get_bsvalues_fast, get_prob):
    with torch.no_grad():
        prob_log = get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))
        origin_prob_log = torch.log(get_prob(get_bsvalues_fast(cur_layer_input, model, final_var))[:, predict_index])
        cur_attn_vector_plus = cur_layer_input.detach().clone()
        for vec in cur_attn_vector_list:
            cur_attn_vector_plus = cur_attn_vector_plus - vec
        cur_attn_vector_bsvalues = get_bsvalues(cur_attn_vector_plus, model, final_var)
        cur_attn_vector_probs = get_prob(cur_attn_vector_bsvalues)
        rank = (torch.argsort(cur_attn_vector_probs, descending=True) == predict_index).nonzero(as_tuple=True)[1]
        cur_attn_vector_probs = cur_attn_vector_probs[:, predict_index]
        cur_attn_vector_probs_log = torch.log(cur_attn_vector_probs)
        cur_attn_vector_probs_log_increase = origin_prob_log - cur_attn_vector_probs_log.cuda()
    return cur_attn_vector_probs_log_increase, torch.exp(cur_attn_vector_probs_log), rank


def get_log_increase_P_vs_N(cur_layer_input, layer_i, all_pos_coefficient_scores, final_var, predict_index, model, get_fc2_params, LAYER_NUM, get_bsvalues, get_prob):
    coefficient_scores = torch.tensor(all_pos_coefficient_scores[layer_i][-1])
    fc2_vectors = get_fc2_params(model, layer_i)
    ffn_subvalues = (coefficient_scores * fc2_vectors).T
    origin_prob_log = torch.log(get_prob(get_bsvalues(cur_layer_input, model, final_var))[predict_index])
    cur_attn_vector_plus = cur_layer_input + ffn_subvalues
    if layer_i == 25:
        import pdb; pdb.set_trace()
    cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(cur_attn_vector_plus, model, final_var))[:, predict_index])
    cur_attn_vector_probs_log_increase = cur_attn_plus_probs - origin_prob_log
    return cur_attn_vector_probs_log_increase


def plt_bar(x, y, yname="log increase", name="Wait"):
    x_major_locator = MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a / 2 for a in x]
    plt.xlim(-0.5, plt_x[-1] + 0.49)
    plt.ylim(-2, 10)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i % 2 == 0:
            x_attn.append(x[i] / 2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i] / 2)
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
                yticklabels=["Attention", "FFN"], vmin=-1, vmax=5)
    plt.xlabel("Layer")
    plt.ylabel("Type")
    plt.title(yname)
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    plt.savefig(f"layer_importance.{safe_name}.png", bbox_inches='tight')
    plt.show()


def plot_bar_heatmap_ffn(data, yname="Value", name="Layer Heatmap"):
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


def plot_bar_heatmap_query_layer_position(curfile_ffn_score_dict, name="query_ffn_heatmap"):
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
    colors = [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
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


def plt_bar_ffn_neuron(x, y, yname="log increase", name="Wait"):
    x_major_locator = MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a / 2 for a in x]
    plt.xlim(-0.5, plt_x[-1] + 0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i % 2 == 0:
            x_attn.append(x[i] / 2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i] / 2)
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


def plt_bar_ffn_range(x, y, yname="log increase", name="Wait", first=-1, second=-1):
    x_major_locator = MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a / 2 for a in x]
    plt.xlim(-0.5, plt_x[-1] + 0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i % 2 == 0:
            x_attn.append(x[i] / 2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i] / 2)
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
    cbar.set_label('Value (|x| < 0.01 -> 0)')
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


def plot_attention_heatmaps(
    all_attn_scores,
    save_dir="attn_heatmaps",
    fold=50,
    source_tokens=None,
    target_tokens=None,
    vmax=None,
    vmin=None
):
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
    print(f"Attention heatmaps saved to {save_dir}")
