import networkx as nx
from utils import write_to_file, graph_details, getTokenizer
from gen_random_graph import create_random_graph
from utils import getMaxEdges

def if_cyclic(G):
    try:
        is_cyclic = nx.find_cycle(G)
        return True
    except nx.NetworkXNoCycle:
        return False
    
def cycle_datasets_generation(config):
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
        # label balance
        pos = 0
        while 1:
            random_graph = create_random_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            cyclic = if_cyclic(random_graph)
            node_nums, edges_flat = graph_details(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            if cyclic:
                ans = config["answer"].format("Yes")
                pos += 1
            else:
                ans = config["answer"].format("No")
            if cyclic and pos > config["samples_needed"][i] / 2:
                continue
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
            