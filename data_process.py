import json
import copy
import os
from tqdm import tqdm

def transform_data(input_dir, dev_num=1000):
    # 将UltraLink格式转为MiniCPM格式
    input_dir = "/data/public/wangshuo/UltraLink/generated_datas/omg-sft"
    output_dir = os.path.join(input_dir, 'minicpm')
    languages = ['en','zh','es','ru','fr']
    for file in os.listdir(input_dir):
        print('current file:', file)
        if not file.endswith(".jsonl"):
            continue
        counter = 0
        data_out_list = []
        with open(os.path.join(input_dir, file), "r") as fi, \
            open(os.path.join(input_dir, 'minicpm' , 'train_' + file), "a") as fw, \
            open(os.path.join(input_dir, 'minicpm' , 'dev_' + file), "a") as fw2:
                for line in tqdm(fi):
                    if len(line.strip()) > 0:
                        data = json.loads(line)
                        ndata = {}
                        ndata['messages'] = []
                        flag_bit = 0
                        for sentence in data['data']:
                            if flag_bit == 0:
                                msg = {
                                    'role': 'user',
                                    'content': copy.deepcopy(sentence)
                                }
                            else:
                                msg = {
                                    'role': 'assistant',
                                    'content': copy.deepcopy(sentence)
                                }
                            ndata['messages'].append(copy.deepcopy(msg))
                            flag_bit = 1 - flag_bit
                        data_out_list.append(ndata)
                        counter += 1
                        if counter == dev_num:
                            json.dump(data_out_list, fw2, ensure_ascii=False, indent=4)
                            data_out_list = []
                json.dump(data_out_list, fw, ensure_ascii=False, indent=4)

def merge_data():
    # 合并MiniCPM格式的数据
    base_path = "/data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/"
    file_list = [
        "en_math",
        "zh_math",
    ]
    outfile = os.path.join("./datas/", '_'.join(file_list) + ".jsonl")
    data=[]
    for file in file_list:
        with open(os.path.join(base_path, "train_" + file + ".jsonl"), "r") as f1:
            data1 = json.load(f1)
            data.extend(data1)
    with open(outfile, "w") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)
            
            
if __name__ == '__main__':
    pass
    