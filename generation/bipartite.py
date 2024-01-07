from networkx.algorithms import bipartite
from utils import write_to_file, getTokenizer, graph_details
from gen_random_graph import create_random_graph, create_random_bipartite_graph

def if_bipartite(G):
    return bipartite.is_bipartite(G)

def bipartite_datasets_generation(config):
    tokenizer = getTokenizer()
    for i in range(len(config["min_nodes"])):
        min_nodes = config["min_nodes"][i]
        max_nodes = config["max_nodes"][i]
        max_edges = config["max_edges"][i]
        min_ratio = config["min_ratio"][i]
        max_ratio = config["max_ratio"][i]
        weight = config["weight"][i]
        directed = config["directed"][i]
        valid = 0
        dup = set()
        while 1:
            random_graph = create_random_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            bi = if_bipartite(random_graph)
            node_nums, edges_flat = graph_details(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            if bi:
                ans = config["answer"].format("Yes")
            else:
                ans = config["answer"].format("No")
            # length check
            if len(tokenizer.encode(input_prompt + ans)) > 3000:
                continue
            sample = {}
            sample["input_prompt"] = input_prompt
            sample["answer"] = ans
            write_to_file(config["store_path"], sample)
            valid += 1
            if valid == config["samples_needed"][i]/2:
                break
        
        valid = 0
        dup = set()
        while 1:
            random_graph = create_random_bipartite_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            bi = if_bipartite(random_graph)
            node_nums, edges_flat = graph_details(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            if bi:
                ans = config["answer"].format("Yes")
            else:
                ans = config["answer"].format("No")
            # length check
            if len(tokenizer.encode(input_prompt + ans)) > 3000:
                continue
            sample = {}
            sample["input_prompt"] = input_prompt
            sample["answer"] = ans
            write_to_file(config["store_path"], sample)
            valid += 1
            if valid == config["samples_needed"][i]/2:
                break