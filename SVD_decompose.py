import os
import argparse
import torch
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_USV(base_path, usv_path):
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_path,torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    # finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_path,torch_dtype=torch.bfloat16,  trust_remote_code=True).to(device)
    # USV乘完之后的是delta
    # usv = torch.load(usv_path)
    # for name, module in finetuned_model.named_modules():
    #     if "self_attn" in name or "mlp" in name:
    #         for subname, submodule in module.named_children():
    #             with torch.no_grad():
    #                 if "proj" in subname:
    #                     # 还需要一个sft后的其它参数
    #                     base = base_model.get_submodule(f"{name}.{subname}").weight.data.clone()
    #                     u = usv[name + '.' + subname + ".U"]
    #                     s = usv[name + '.' + subname + ".S"]
    #                     v = usv[name + '.' + subname + ".V"]
    #                     #delta是和base的差
    #                     delta = torch.matmul(torch.matmul(u,torch.diag(s)),v.t())
    #                     finetuned_model.get_submodule(f"{name}.{subname}").weight.copy_(base + delta)
          
          
    finetuned_model = base_model              
    usv = torch.load(usv_path)
    for name, module in tqdm(finetuned_model.named_modules()):
        if "self_attn" in name or "mlp" in name:
            for subname, submodule in module.named_children():
                with torch.no_grad():
                    if "proj" in subname:
                        # 还需要一个sft后的其它参数
                        u = usv[name + '.' + subname + ".U"]
                        s = usv[name + '.' + subname + ".S"]
                        v = usv[name + '.' + subname + ".V"]
                        #delta是和base的差
                        delta = torch.matmul(torch.matmul(u,torch.diag(s)),v.t())
                        finetuned_model.get_submodule(f"{name}.{subname}").weight.add_(delta)
                        
    return tokenizer, finetuned_model

def decomposition(masked_input_tensor,dim=None):
    U , S , V = torch.svd(masked_input_tensor.to(torch.float32))
    
    outlier_U , outlier_V = None, None
    
    if dim is not None:
        U , S , V = U[:, :dim],S[:dim] ,V[:, :dim]
    
    return U, S, V 


def SVD_decompose(base_model,finetuned_model,dim_attn,save_path):
    base_model = AutoModelForCausalLM.from_pretrained(base_model,torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model,torch_dtype=torch.bfloat16,  trust_remote_code=True).to(device)
    
    param_dict = dict()
    for k,v in tqdm(base_model.state_dict().items()):
        if "self_attn" in k or "mlp" in k:
            if ".weight" in k:
                #delta需要在1e-4级别以下
                delta = finetuned_model.state_dict()[k] - v
                dim = dim_attn
                # pdb.set_trace()
                # #是否需要放缩
                # if "mlp" in k:
                #     dim = int(dim * 1.45)
                U,S,V = decomposition(delta, dim=dim)
                
                k = k.replace(".weight", "")
                
                param_dict[k + ".base"] = v
                param_dict[k + ".U"] = U.data.to(torch.bfloat16)
                param_dict[k + ".S"] = S.data.to(torch.bfloat16)
                param_dict[k + ".V"] = V.data.to(torch.bfloat16)
            
                # import pdb; pdb.set_trace()
            
    torch.save(param_dict, save_path)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_svd', action='store_true', help='generate SVD lora')
    parser.add_argument('--merge', action='store_true', help='merge SVD lora into base model')
    parser.add_argument('--dim', type=int, default=64, help='SVD lora dimension')
    parser.add_argument('--base_model', type=str, default="", help='pretrained model path')
    parser.add_argument('--finetuned_model', type=str, default="", help='finetuned model path')
    parser.add_argument('--usv_path', type=str, default="", help='SVD lora path')
    parser.add_argument('--save_path', type=str, default="", help='output path')    
    args = parser.parse_args()
    
    if args.use_svd:
        print("Using SVD to decompose the delta")
        base_model = args.base_model
        if base_model == "":
            print("Please specify the base model")
            exit()
        finetuned_model = args.finetuned_model
        if finetuned_model == "":
            print("Please specify the finetuned model")
            exit()
        dim = args.dim
        save_path = args.save_path
        if save_path == "":
            print("Please specify the save path")
            exit()
        
        print("SVD decomposition")
        print(f"base model: {base_model}")
        print(f"finetuned model: {finetuned_model}")
        print(f"save path: {save_path}")
        print(f"dim: {dim}")
        
        SVD_decompose(base_model=base_model,finetuned_model=finetuned_model,dim_attn=dim,save_path=save_path)
    elif args.merge:
        base_model = args.base_model
        if base_model == "":
            print("Please specify the base model")
            exit()
        usv_path = args.usv_path
        if usv_path == "":
            print("Please specify the usv path")
            exit()
        save_path = args.save_path
        if save_path == "":
            print("Please specify the save path")
            exit()
        
        print("Merge the base model and the delta")
        print(f"base model: {base_model}")
        print(f"usv path: {usv_path}")
        print(f"save path: {save_path}")
        
        tokenizer , model = load_USV(base_path=base_model, finetuned_path=base_model, usv_path=usv_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
    else:
        print("Please specify the operation")