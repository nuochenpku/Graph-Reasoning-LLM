import os
import json
import re
from typing import Optional, Dict, Sequence, List
from methods import filter_string, sorted_distance, sorted_jaccard, sorted_tfidf, TextModel
from pathlib import Path
import json
import tqdm
import torch
from sklearn.cluster import KMeans
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(description="Split a JSON file into odd and even lines.")
parser.add_argument('--task', type=str, default="cycle", help="The task name for the output file.")
parser.add_argument('--text_encoder', type=str, default="SentenceBert", help="The task name for the output file.")
parser.add_argument('--k', type=int, default=3, help="k-means")
parser.add_argument('--device', type=int, default=0, help="gpu device")
parser.add_argument('--npz_path', type=str, default="../datasets/embeddings", help="npz path")
parser.add_argument('--save_path', type=str, default="../datasets/rft_dpo_dicts", help="save path")
args = parser.parse_args()


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  
    res = re.findall(r"(\d+(\.\d+)?)", text) 
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
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
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


## define a function that read the different task files where each task has 20 json files with indicing from 0 to 20
def align_json_files(task_dicts: Dict, folder_path: str, task: str):
    
    # aligned_list = []
    aligned_dict = task_dicts
    file_paths = glob.glob(os.path.join(folder_path, f"*{task}*.jsonl"))
    for file_path in file_paths:
        # file_path = os.path.join(folder_path, f"{file_index}_gen_{task}_datas.jsonl")
        print(f"Processing file: {file_path}")
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
                    
                    if check(task, json_data['answer'], response) and len(response.split()) > 20:
                        aligned_dict[query]['pos_response'].append(response)
                        
                    elif not check(task, json_data['answer'], response):
                        aligned_dict[query]['neg_response'].append(response)
                        
                    aligned_dict[query]['task'] = task
                    
                else:
                    if check(task, json_data['answer'], response)  and len(response.split()) > 20:
                        aligned_dict[query]['pos_response'].append(response)
                        
                    elif not check(task, json_data['answer'], response):
                        aligned_dict[query]['neg_response'].append(response)
    
    for key, value in aligned_dict.items():
        aligned_dict[key]['neg_response'] = list(set(aligned_dict[key]['neg_response']))    
        aligned_dict[key]['pos_response'] = list(set(aligned_dict[key]['pos_response']))    
        
    return aligned_dict
               
def CoT_response(task):
    with open('/home/yuhanli/Graph-Reasoning-LLM/datasets/data/graph_source_data_v1_dsformat.json') as  f:
        datas = f.readlines()
    
    task_dicts = {}
    for data in datas:
        data = json.loads(data)
        query = data['query'].replace('\n', '')
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

def compute_cosine_similarity(truth_emb, path_embs):
    truth_norm = truth_emb.norm(dim=1, keepdim=True)
    path_norms = path_embs.norm(dim=1, keepdim=True)
    
    truth_emb = truth_emb / truth_norm
    path_embs = path_embs / path_norms

    similarity = torch.mm(path_embs, truth_emb.transpose(0, 1))
    
    similarity = similarity.squeeze().cpu().numpy()
    return similarity

def sorted_cosine(model, paths, truth, type):
    path_embs = text_emb(model, paths, type + "_paths")
    truth_emb = text_emb(model, [truth], type + "_truth")
    cos_similarities = compute_cosine_similarity(truth_emb, path_embs)
    sorted_indices = np.argsort(cos_similarities)[::-1].tolist()
    return sorted_indices, truth_emb, path_embs

def get_string_similarity_index(task_dicts):
    
    # diver_dicts = {}
    # neg_dicts = {}
    # pos_dicts = {}
    rft_dpo_dicts = {}

    text_model = TextModel(args.text_encoder)

    embedding_dict = {}

    for key, value in task_dicts.items():
        if 'CoT_response' in value:
            truth = filter_string(value['CoT_response'])
            if len(value['pos_response']) >=1:
                print("pos: " + str(len(value['pos_response'])) + " neg: " + str(len(value['neg_response'])))
                dist_indic = sorted_distance(value['pos_response'], truth)
                jaccard = sorted_jaccard(value['pos_response'], truth)
                tfidf = sorted_tfidf(value['pos_response'], truth)
                # cosine similarity
                semantic_cosine, truth_emb, path_embs = sorted_cosine(text_model, value['pos_response'], truth, "pos")
                # save the truth embedding and path embeddings
                if key in embedding_dict:
                    embedding_dict[key]['pos_path_embs'] = path_embs.cpu().numpy()
                    embedding_dict[key]['truth_emb'] = truth_emb.cpu().numpy()
                else:
                    embedding_dict[key] = {
                        'truth_emb': truth_emb.cpu().numpy(),
                        'pos_path_embs': path_embs.cpu().numpy(),
                    }
                pos_sort = {
                    'edit': dist_indic,
                    'jaccard': jaccard,
                    'tfidf': tfidf,
                    'cosine': semantic_cosine
                }
                value['pos_sort'] = pos_sort
            
            if len(value['neg_response']) >=1:
                dist_indic = sorted_distance(value['neg_response'], truth)
                jaccard = sorted_jaccard(value['neg_response'], truth)
                tfidf = sorted_tfidf(value['neg_response'], truth)
                # cosine similarity
                semantic_cosine, truth_emb, path_embs = sorted_cosine(text_model, value['neg_response'], truth, "neg")
                # save the truth embedding and path embeddings
                if key in embedding_dict:
                    embedding_dict[key]['neg_path_embs'] = path_embs.cpu().numpy()
                    embedding_dict[key]['truth_emb'] = truth_emb.cpu().numpy()
                else:
                    embedding_dict[key] = {
                        'truth_emb': truth_emb.cpu().numpy(),
                        'neg_path_embs': path_embs.cpu().numpy(),
                    }
                neg_sort = {
                    'edit': dist_indic,
                    'jaccard': jaccard,
                    'tfidf': tfidf,
                    'cosine': semantic_cosine
                }
                value['neg_sort'] = neg_sort
        # little change here. we only need to return samples which have at least one positive response.
        if 'pos_sort' in value:
            rft_dpo_dicts[key] = value

    save_path = os.path.join(args.npz_path, f"{args.task}_{args.text_encoder}_embeddings.npz")
    np.savez_compressed(save_path, **embedding_dict)
    print(f"All embeddings saved to {save_path}")


    return rft_dpo_dicts    
    
        # if 'neg_sort' in value and 'pos_sort' in value:
        #     diver_dicts[key] = value
            
        # elif 'neg_sort' in value and 'pos_sort' not in value:
        #     neg_dicts[key] = value
            
        # elif 'neg_sort' not in value and 'pos_sort' in value:
        #     pos_dicts[key] = value
            
    # return diver_dicts, neg_dicts, pos_dicts


def compute_and_sort_length(lst: List[str], string: str=None) -> List[str]:
    split_lengths = [(element, len(element.split('\n'))) for element in lst]
    sorted_lengths = sorted(split_lengths, key=lambda x: x[1])
    if string:
        sorted_strings = [element[0] for element in sorted_lengths if element[1] > len(string.split('\n'))]
    else:
        sorted_strings = [element[0] for element in sorted_lengths]
    print(sorted_strings)
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
        
    if positive:
        rft_paths.append(sample['pos_sort']['edit'][-1])
        rft_paths.append(sample['pos_sort']['jaccard'][-1])
        rft_paths.append(sample['pos_sort']['tfidf'][-1])
        rft_paths.append(sample['pos_sort']['cosine'][-1])
    else:
        rft_paths.append(sample['neg_sort']['edit'][0])
        rft_paths.append(sample['neg_sort']['jaccard'][0])
        rft_paths.append(sample['neg_sort']['tfidf'][0])
        rft_paths.append(sample['neg_sort']['cosine'][0])
    
    return rft_paths
  
  
def text_emb(model, paths, type):
    model = model.to(args.device)
    text_features = []
    for text in tqdm.tqdm(paths, desc="Processing" + " " + type):
        text_features.append(model(text).unsqueeze(dim=0).cpu())
    features = torch.cat(text_features, dim=0)
    return features    

# k: 2, 3, 4
def kmeans(text_embs, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(text_embs)
    centers = kmeans.cluster_centers_
    center_indices = []
    for center in centers:
        distances = np.linalg.norm(text_embs - center, axis=1)
        center_indices.append(np.argmin(distances))
    return center_indices

   
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
            if len(sample['pos_response']) >= 1:
                pos_rft_paths = align_rft_paths(sample, positive=True)
            # pos_rft_paths = align_rft_paths(sample, positive=True)
            if len(sample['neg_response']) >= 1:
                neg_rft_paths = align_rft_paths(sample, positive=False)
        else:
            # if positive:
            if len(sample['pos_response']) >= 1:
                pos_rft_paths = compute_and_sort_length(sample['pos_response'], sample['CoT_response'])
            # else:
            if len(sample['neg_response']) >= 1:
                neg_rft_paths = compute_and_sort_length(sample['neg_response'], sample['CoT_response'])

    sample['pos_rft_paths'] =  pos_rft_paths 
    sample['neg_rft_paths'] =  neg_rft_paths 
    return sample        


def main():

    # tasks = ['connectivity', 'cycle','flow', 'bipartite', 'hamilton', 'triplet', 'shortest', 'topology', 'substructure',]
    
    if os.path.exists(Path(args.save_path) / f"{args.task}_rft_dpo_dicts.json"):
        with open(Path(args.save_path) / f"{args.task}_rft_dpo_dicts.json", "r") as f:
            rft_dpo_dicts = json.load(f)
    else:
        task_dicts = CoT_response(args.task)

        folder_path = "/home/yuhanli/Graph-Reasoning-LLM/datasets/xrft"
        task_aligned_dicts = align_json_files(task_dicts, folder_path, task=args.task)
        
        rft_dpo_dicts = get_string_similarity_index(task_aligned_dicts)
        
        # save rft_dpo_dicts
        with open(Path(args.save_path) / f"{args.task}_rft_dpo_dicts.json", "w", encoding='utf-8') as f:
            json.dump(rft_dpo_dicts, f, ensure_ascii=False, indent=4)
    
    for query, value in rft_dpo_dicts.items():
        new_sample = merge_rft_paths(value) 
        print(query, new_sample)

if __name__ == "__main__":
    main()