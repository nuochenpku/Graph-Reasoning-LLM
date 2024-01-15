import os
import json
import re
from typing import Optional, Dict, Sequence, List
from methods import filter_string, sorted_distance, sorted_jaccard, sorted_tfidf
from pathlib import Path
import json


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
    
    
def check(key, truth, predict):
    
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                # correct_samples[key].append(v)
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'(yes|no)', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            else:
                return False
    elif key in ['flow', 'shortest', 'triplet']:
      
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        if abs(t_num - p_num) < 1e-2:
            return True
        return False
                
    elif key == 'topology':
        
        # elif key == 'topology':
        
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth = truth.split('###')[-1].strip(' ')
            if truth in pre or pre in truth:
                return True
            return False
        else:
            truth = truth.split('###')[-1].split(',')
            for t in truth:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False


## define a function that read the different task files where each task has 20 json files with indicing from 0 to 20def read_task_files(folder_path):
def align_json_files(task_dicts: Dict, folder_path: str, task: str):
    
    # aligned_list = []
    aligned_dict = task_dicts
    for file_index in range(21):
        file_path = os.path.join(folder_path, f"{file_index}_gen_{task}_datas.jsonl")
        
        with open(file_path, "r") as file:
            data = file.readlines()

            for line in data:
                data = json.loads(line)
                json_data = data['source_data']
                query = json_data["input_prompt"]
                response = data["output_str"]
                
                if query not in aligned_dict:
                    aligned_dict[query] = dict()
                    aligned_dict[query]['neg_response'] = []
                    aligned_dict[query]['pos_response'] = []
                    
                    if check(task, json_data['answer'], response) and len(response) >15:
                        aligned_dict[query]['pos_response'].append(response)
                        
                    elif not check(task, json_data['answer'], response):
                        aligned_dict[query]['neg_response'].append(response)
                        
                    aligned_dict[query]['task'] = task
                    
                else:
                    if check(task, json_data['answer'], response)  and len(response) >15:
                        aligned_dict[query]['pos_response'].append(response)
                        
                    elif not check(task, json_data['answer'], response):
                        aligned_dict[query]['neg_response'].append(response)
    
    for key, value in aligned_dict.items():
        aligned_dict[key]['neg_response'] = list(set(aligned_dict[key]['neg_response']))    
        aligned_dict[key]['pos_response'] = list(set(aligned_dict[key]['pos_response']))    
        
    return aligned_dict
               
def CoT_response(task):
    with open('/cpfs/user/chennuo/CN/Graph-Reasoning-LLM/datasets/data/graph_source_data_v1_dsformat.json') as  f:
        datas = f.readlines()
    
    task_dicts = {}
    for data in datas:
        data = json.loads(data)
        query = data['query']
        if task == data['task']:
            task_dict = {
                "CoT_response": data["CoT_response"],
                "answer": data["response"],
                'task': data['task'],
                'pos_response': [],
                'neg_response': []
            }
            task_dicts[query] = task_dict
    return task_dicts


def find_path(task_dicts):
    
    diver_dicts = {}
    neg_dicts = {}
    pos_dicts = {}
    
    for key, value in task_dicts.items():
        
        if 'CoT_response' in value:
            
            truth = filter_string(value['CoT_response'])
            
            if len(value['pos_response']) >=1:
                
                dist_indic = sorted_distance(value['pos_response'], truth)
                jaccard = sorted_jaccard(value['pos_response'], truth)
                tfidf = sorted_tfidf(value['pos_response'], truth)
                
                pos_sort = {
                    'edit': dist_indic,
                    'jaccard': jaccard,
                    'tfidf': tfidf
                }
                value['pos_sort'] = pos_sort
            
            if len(value['neg_response']) >=1:
                
                dist_indic = sorted_distance(value['neg_response'], truth)
                jaccard = sorted_jaccard(value['neg_response'], truth)
                tfidf = sorted_tfidf(value['neg_response'], truth)

                neg_sort = {
                    'edit': dist_indic,
                    'jaccard': jaccard,
                    'tfidf': tfidf
                }
                value['neg_sort'] = neg_sort
                
        ### add k-means and cosine similarity
        ### TO-DO
        
        ### collect as final data dicts:
        if 'neg_sort' in value and 'pos_sort' in value:
            diver_dicts[key] = value
            
        elif 'neg_sort' in value and 'pos_sort' not in value:
            neg_dicts[key] = value
            
        elif 'neg_sort' not in value and 'pos_sort' in value:
            pos_dicts[key] = value
            
    return diver_dicts, neg_dicts, pos_dicts



def compute_and_sort_length(lst: List[str], string: str=None) -> List[str]:
    split_lengths = [(element, len(element.split('\n'))) for element in lst]
    sorted_lengths = sorted(split_lengths, key=lambda x: x[1])
    if string:
        sorted_strings = [element[0] for element in sorted_lengths if element[1] > len(string.split('\n'))]
    else:
        sorted_strings = [element[0] for element in sorted_lengths]
    return sorted_strings




def parse_triplet_shortest(lst: List[str], string: str=None):
    pattern = r'<<([^>]*)>>' 
    exist_matches = []
    rft_paths = []
    try:
        if string:
            matches = re.findall(pattern, string)
            matches = '|'.join(matches).replace(' ', '')
            exist_matches.append(matches)
            rft_paths.append(string)
            
        for q in lst:
            matches = re.findall(pattern, q)  # Find all matches
            matches = '|'.join(matches).replace(' ', '')
            if matches not in exist_matches:
                exist_matches.append(matches)
                rft_paths.append(q)
        
        return rft_paths
    
    except BaseException:
        return  None
   
def parse_topology(lst: List[str], string: str=None):
    pattern = r'\[(.*?)\]' 
    exist_matches = []
    rft_paths = []
    try:
        if string:
            matches = re.findall(pattern, string)
            matches = '|'.join(matches).replace(' ', '')
            exist_matches.append(matches)
            rft_paths.append(string)
            
        for q in lst:
            matches = re.findall(pattern, q)  # Find all matches
            matches = '|'.join(matches).replace(' ', '')
            if matches not in exist_matches:
                exist_matches.append(matches)
                rft_paths.append(q)
        
        return rft_paths
    
    except BaseException:
        return  None
        
  
def align_rft_paths(sample: Dict, positive: bool=True):
    
    rft_paths = []
    
    if positive:
        len_rft_paths = compute_and_sort_length(sample['pos_response'], sample['CoT_response'])
    else:
        len_rft_paths = compute_and_sort_length(sample['neg_response'], sample['CoT_response'])
        
    if len_rft_paths != []:
        rft_paths.append(len_rft_paths[-2:])
        
    rft_paths.append(sample['neg_sort']['edit'][-1])
    rft_paths.append(sample['neg_sort']['jaccard'][-1])
    rft_paths.append(sample['neg_sort']['tfidf'][-1])
            
            
    ## add embedding $ k-means
    ### TO-DO
    
    return list(set(rft_paths)) 
  
  
        
    
   
def merge_rft_paths(sample: Dict):
    
    task = sample['task']
    
    if task == 'topology':
        
        if 'CoT_response' in sample:
            
            pos_rft_paths = parse_topology(sample['pos_response'], sample['CoT_response'])
            neg_rft_paths = parse_topology(sample['neg_response'], sample['CoT_response'])
        else:
            pos_rft_paths = parse_topology(sample['pos_response'])
            neg_rft_paths = parse_topology(sample['neg_response'])
        
    elif task in ['triplet', 'shortest']:
        
        if 'CoT_response' in sample:
            
            pos_rft_paths = parse_triplet_shortest(sample['pos_response'], sample['CoT_response'])
            neg_rft_paths = parse_triplet_shortest(sample['neg_response'], sample['CoT_response'])
        else:
            pos_rft_paths = parse_triplet_shortest(sample['pos_response'])
            neg_rft_paths = parse_triplet_shortest(sample['neg_response'])
    else:
        if 'CoT_response' in sample:
            
            pos_rft_paths = align_rft_paths(sample)
            neg_rft_paths = align_rft_paths(sample)
            
        else:
            
            # if positive:
            len_rft_paths = compute_and_sort_length(sample['pos_response'], sample['CoT_response'])
        # else:
            len_rft_paths = compute_and_sort_length(sample['neg_response'], sample['CoT_response'])
            
            
            ####add embedding & cosine
            ### TO-DO
        
    sample['pos_rft_paths'] =  pos_rft_paths 
    sample['neg_rft_paths'] =  neg_rft_paths 
    return sample        
        
    
if __name__ == 'main':
    
    tasks = ['connectivity', 'cycle','flow', 'bipartite', 'hamilton', 'triplet', 'shortest', 'topology', 'substructure',]
    
    diverse_dir = ''

    
    for task in tasks:
    
        task_dicts = CoT_response(task)
    
        task_aligned_dicts = align_json_files(task_dicts, folder_path='', task=task)
        
        ### diver_dicts has pos and negs, neg_dicts only negs, pos_dicts only poss
        diver_dicts, neg_dicts, pos_dicts = find_path(task_aligned_dicts)
        
        with open(Path(diverse_dir) / f"only_diverse_paths.json", "w", encoding='utf-8') as f:
            json.dump(diver_dicts, f, ensure_ascii=False, indent=4)
        
        for query, value in diver_dicts:
            new_sample = merge_rft_paths(value)
        
        
        