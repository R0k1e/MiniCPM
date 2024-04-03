import argparse
import json
import os
import shutil
from tqdm import tqdm
from collections import OrderedDict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from convert_hf_to_vllmcpm import convert_model

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
                    print(f"Converting {load_path}...")
                    if load_path.endswith('-vllm'):
                        continue
                    if os.path.exists(load_path + '-vllm'):
                        continue
                    save_path = load_path + '-vllm'
                    load_model_ckpt(load_path, save_path)  