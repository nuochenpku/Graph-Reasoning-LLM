import networkx as nx
from utils import write_to_file, getTokenizer, graph_details, getMaxEdges
from gen_random_graph import create_random_graph

def topological_sort(G):
    try:
        flag = nx.is_directed_acyclic_graph(G)
        return list(nx.all_topological_sorts(G))
    except nx.NetworkXUnfeasible:
        return list()
    
def topology_datasets_generation(config):
    tokenizer = getTokenizer()
    index = 0
    for i in range(len(config["min_nodes"])):
        min_nodes = config["min_nodes"][i]
        max_nodes = config["max_nodes"][i]
        max_edges = config["max_edges"][i]
        min_ratio = config["min_ratio"][i]
        max_ratio = config["max_ratio"][i]
        weight = config["weight"][i]
        directed = config["directed"][i]
        valid = 0
        edges_number = [int(getMaxEdges(min_nodes) * min_ratio), int(getMaxEdges(max_nodes) * max_ratio)]
        nodes_number = [min_nodes, max_nodes]
        dup = set()
        # test duplicate check
        if "test" in config["store_path"]:
            # read from train
            with open(config["store_path"].replace("test", "train"), "r") as f:
                for line in f:
                    if line.strip() == "":
                        continue
                    sample = eval(line.strip())
                    dup.add(sample["input_prompt"])
        while 1:
            random_graph = create_random_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            topology_paths = topological_sort(random_graph)
            node_nums, edges_flat = graph_details(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            ans = config["answer"].format(str(topology_paths))
            # length check
            if len(tokenizer.encode(input_prompt + ans)) > 3000:
                continue
            sample = {}
            sample["index"] = index
            index += 1
            sample["input_prompt"] = input_prompt
            sample["answer"] = ans
            sample["node_range"] = nodes_number
            sample["edge_range"] = edges_number
            write_to_file(config["store_path"], sample)
            valid += 1
            if valid == config["samples_needed"][i]:
                break