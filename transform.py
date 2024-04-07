import json
import copy
import os
from tqdm import tqdm

input_dir = "/data/public/wangshuo/UltraLink/generated_datas/omg-sft"
output_dir = os.path.join(input_dir, 'minicpm')
languages = ['en','zh','es','ru','fr']

def transform_data(input_dir, output_dir):
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
                        if counter == 1000:
                            json.dump(data_out_list, fw2, ensure_ascii=False, indent=4)
                            data_out_list = []
                json.dump(data_out_list, fw, ensure_ascii=False, indent=4)

def merge_data(input_dir, output_dir):
    train_datas = {
            'en': [],
            'zh': [],
            'es': [],
            'ru': [],
            'fr': []
        }
    eval_datas = {
            'en': [],
            'zh': [],
            'es': [],
            'ru': [],
            'fr': []
        }
    for file in os.listdir(output_dir):
        print('current file:', file)
        if not file.endswith(".jsonl"):
            continue
        
        names = file.split('_')
        lang = names[1]
        Type = names[0]

        if lang not in languages:
            continue
        
        with open(os.path.join(output_dir, file), "r") as fi:
            messages = copy.deepcopy(json.load(fi))
            if Type == 'train':
                train_datas[lang].extend(messages)
            else:
                eval_datas[lang].extend(messages)

    for lang, train_dataset in train_datas.items():
        file_name = 'train_' + lang + '_all.jsonl'
        with open(os.path.join(output_dir, file_name), "w") as fw:
            json.dump(train_datas[lang], fw, ensure_ascii=False, indent=4)

    for lang, eval_dataset in eval_datas.items():
        file_name = 'dev_' + lang + '_all.jsonl'
        with open(os.path.join(output_dir, file_name), "w") as fw:
            json.dump(eval_datas[lang], fw, ensure_ascii=False, indent=4)

def merge_chat_data(input_dir, output_dir):
    train_chat_data = {
        'en': [],
        'zh': [],
        'es': [],
        'ru': [],
        'fr': []
    }

    eval_chat_data = {
        'en': [],
        'zh': [],
        'es': [],
        'ru': [],
        'fr': []
    }

    for file in os.listdir(output_dir):
        print('current file:', file)
        parts = file.split('.')[0].split('_')
        if len(parts) == 3:
            continue
        elif len(parts) == 4:
            Type = parts[0]
            lang = parts[1]
            category = parts[2]

            with open(os.path.join(output_dir, file), "r") as fi:
                messages = copy.deepcopy(json.load(fi))
                if Type == 'train':
                    train_chat_data[lang].extend(messages)
                elif Type == 'dev':
                    eval_chat_data[lang].extend(messages)

    for lang, train_dataset in train_chat_data.items():
        file_name = 'train_' + lang + '_chat_all.jsonl'
        with open(os.path.join(output_dir, file_name), "w") as fw:
            json.dump(train_chat_data[lang], fw, ensure_ascii=False, indent=4)

    for lang, eval_dataset in eval_chat_data.items():
        file_name = 'dev_' + lang + '_chat_all.jsonl'
        with open(os.path.join(output_dir, file_name), "w") as fw:
            json.dump(eval_chat_data[lang], fw, ensure_ascii=False, indent=4)    
            
            
if __name__ == '__main__':
        
    base_path = "/data/public/wangshuo/UltraLink/generated_datas/omg-sft/minicpm/"
    file_list = [
        "ru_all",
        "zh_code",
        "fr_code",
        "es_code"
    ]
    outfile = os.path.join("./datas/", '_'.join(file_list) + ".jsonl")
    data=[]
    for file in file_list:
        with open(os.path.join(base_path, "train_" + file + ".jsonl"), "r") as f1:
            data1 = json.load(f1)
            data.extend(data1)
    with open(outfile, "w") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=4)