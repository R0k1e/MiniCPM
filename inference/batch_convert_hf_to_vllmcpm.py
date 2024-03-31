import argparse
import json
import os
import shutil
from tqdm import tqdm
from collections import OrderedDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def convert_model(config, ckpt):
    # config
    config_bmt = OrderedDict(
        {
            "_dtype": "bf16",
            "activate_fn": "silu",
            "architectures": [
                "CPMDragonflyForCausalLM"
            ],
            "model_type": "cpm_dragonfly",
            "base": 10000,
            "dim_ff": config['intermediate_size'],
            "dim_head": config['hidden_size'] // config['num_attention_heads'],
            "dim_model": config['hidden_size'],
            "dim_model_base": 256,
            "dropout_p": 0.0,
            "eps": config['rms_norm_eps'],
            "init_std": config['initializer_range'],
            "num_heads": config['num_attention_heads'],
            "num_kv_heads": config['num_key_value_heads'],
            "num_layers": config['num_hidden_layers'],
            "orig_max_length": 4096,
            "pose_prob": 0.0,
            "pose_scaling_factor": 1.0,
            "qk_norm": False,
            "rope_scaling_factor": 1,
            "rope_scaling_type": "",
            "scale": True,
            "scale_depth": config['scale_depth'],
            "scale_emb": config['scale_emb'],
            "tie_lm_head": True,
            "tp": 0,
            "transformers_version": "4.35.0",
            "vocab_size": config['vocab_size']
        }
    )


    model_bmt = OrderedDict()
    model_bmt["input_embedding.weight"] = ckpt['model.embed_tokens.weight'].contiguous()
    model_bmt["encoder.output_layernorm.weight"] = ckpt['model.norm.weight'].contiguous()
    for lnum in tqdm(range(config_bmt['num_layers'])):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        model_bmt[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"] = ckpt[f"{hf_pfx}.input_layernorm.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_q.weight"] = ckpt[f"{hf_pfx}.self_attn.q_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_k.weight"] = ckpt[f"{hf_pfx}.self_attn.k_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.project_v.weight"] = ckpt[f"{hf_pfx}.self_attn.v_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"] = ckpt[f"{hf_pfx}.self_attn.o_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"] = ckpt[f"{hf_pfx}.post_attention_layernorm.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"] = ckpt[f"{hf_pfx}.mlp.gate_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"] = ckpt[f"{hf_pfx}.mlp.up_proj.weight"].contiguous()
        model_bmt[f"{bmt_pfx}.ffn.ffn.w_out.weight"] = ckpt[f"{hf_pfx}.mlp.down_proj.weight"].contiguous()


    return config_bmt, model_bmt

def load_model_ckpt(load_path, save_path):
    with open(os.path.join(load_path, "config.json"), 'r') as fin:
        config = json.load(fin)
    # ckpt = torch.load(os.path.join(load_path, "pytorch_model.bin"))
    
    ckpt = AutoModelForCausalLM.from_pretrained(load_path, trust_remote_code=True).state_dict()

    os.makedirs(f"{save_path}", exist_ok=True)

    # model and config
    hf_config, hf_ckpt = convert_model(config, ckpt)
    with open(os.path.join(save_path, "config.json"), 'w') as fout:
        json.dump(hf_config, fout, indent=4)
    torch.save(hf_ckpt, f"{save_path}/pytorch_model.pt")

    # tokenizer
    shutil.copyfile(f"{load_path}/tokenizer.json", f"{save_path}/tokenizer.json")
    # tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    # tokenizer.save_pretrained(os.path.join(save_path, "tokenizer.model"))
    # shutil.copyfile(f"{load_path}/tokenizer.model", f"{save_path}/tokenizer.model")
    shutil.copyfile(f"{load_path}/special_tokens_map.json", f"{save_path}/special_tokens_map.json")
    shutil.copyfile(f"{load_path}/tokenizer_config.json", f"{save_path}/tokenizer_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default="")
    args = parser.parse_args()
    
    load_dir = args.load
    
    for lang_type in os.listdir(load_dir):
        for tags in os.listdir(os.path.join(load_dir, lang_type)):
            for timestamp in os.listdir(os.path.join(load_dir, lang_type, tags)):
                for ckpt_path in os.listdir(os.path.join(load_dir, lang_type, tags,timestamp)):
                    load_path = os.path.join(load_dir, lang_type, tags, timestamp, ckpt_path)
                    if load_path.endswith('-vllm'):
                        continue
                    if os.path.exists(load_path + '-vllm'):
                        continue
                    save_path = load_path + '-vllm'
                    load_model_ckpt(load_path, save_path)  