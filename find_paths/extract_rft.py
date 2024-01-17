import json
import argparse
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import copy

parser = argparse.ArgumentParser(description="Split a JSON file into odd and even lines.")
parser.add_argument('--task', type=str, default="cycle", help="The task name for the output file.")
parser.add_argument('--text_encoder', type=str, default="SentenceBert", help="The task name for the output file.")
parser.add_argument('--k', type=int, default=3, help="k-means")
parser.add_argument('--device', type=int, default=0, help="gpu device")
parser.add_argument('--npz_path', type=str, default="../datasets/embeddings", help="npz path")
parser.add_argument('--dict_path', type=str, default="../datasets/rft_dpo_dicts", help="dict path")
parser.add_argument('--save_path', type=str, default="../datasets/rft_files", help="save path")
args = parser.parse_args()

# \n, 

def CoT_contained_path_extraction(rft_dict, npz_file):
    save_path = Path(args.save_path) / f"{args.task}_rft_paths.json"
    for key, value in rft_dict.items():
        if len(value["pos_response"]) < args.k:
            continue
        sample = {}
        sample["query"] = key
        # string similarity, \n, cosine, k-means
        if "CoT_response" in value:
            # edit
            last_k_indices = value["pos_sort"]["edit"][-args.k:]
            edit_strings = [value["pos_response"][i] for i in last_k_indices]
            sample["edit"] = edit_strings
            # cosine
            last_k_indices = value["pos_sort"]["cosine"][-args.k:]
            cosine_strings = [value["pos_response"][i] for i in last_k_indices]
            sample["cosine"] = cosine_strings
            # jaccard
            last_k_indices = value["pos_sort"]["jaccard"][-args.k:]
            jaccard_strings = [value["pos_response"][i] for i in last_k_indices]
            sample["jaccard"] = jaccard_strings
            # tfidf
            last_k_indices = value["pos_sort"]["tfidf"][-args.k:]
            tfidf_strings = [value["pos_response"][i] for i in last_k_indices]
            sample["tfidf"] = tfidf_strings
            # \n
            CoT_pos_paths = copy.deepcopy(value["pos_response"])
            CoT_pos_paths.append(value["CoT_response"])
            lines_strings = get_lines_ranking(CoT_pos_paths)
            sample["lines"] = lines_strings
            # k-means
            truth_emb = npz_file[key]['truth_emb']
            pos_path_embs = npz_file[key]['pos_path_embs']
            combined_embs = np.vstack((pos_path_embs, truth_emb))
            kmeans_index = kmeans(combined_embs)
            sample["kmeans"] = [CoT_pos_paths[i] for i in kmeans_index]
        else:
            # cannot perform string/cosine
            # \n
            lines_strings = get_lines_ranking(value["pos_response"])
            sample["lines"] = lines_strings
            # k-means
            kmeans_index = kmeans(npz_file[key]['pos_path_embs'])
            sample["kmeans"] = [CoT_pos_paths[i] for i in kmeans_index]

        write_to_file(save_path, sample)


def kmeans(embs):
    kmeans = KMeans(n_clusters=args.k, random_state=42).fit(embs)
    cluster_centers = kmeans.cluster_centers_
    closest_data_points = []
    for center in cluster_centers:
        distances = np.linalg.norm(embs - center, axis=1)
        closest_data_point_index = np.argmin(distances)
        closest_data_points.append(closest_data_point_index)
    return closest_data_points


def get_lines_ranking(CoT_pos_paths):
    sorted_paths = sorted(CoT_pos_paths, key=lambda s: s.count('\n'), reverse=True)
    top_k_paths_with_most_newlines = sorted_paths[:args.k]
    return top_k_paths_with_most_newlines


def write_to_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str)
        f_converted.write('\n')


def main():

    with open(Path(args.dict_path) / f"{args.task}_rft_dpo_dicts.json", "r") as f:
        rft_dict = json.load(f)

    npz_file = np.load(os.path.join(args.npz_path, f"{args.task}_{args.text_encoder}_embeddings.npz"), allow_pickle = True)
    
    CoT_contained_path_extraction(rft_dict, npz_file) 

    npz_file.close()


if __name__ == "__main__":
    main()