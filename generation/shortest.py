import networkx as nx
from utils import write_to_file, getTokenizer, graph_details_with_weight
from gen_random_graph import create_random_graph
from connectivity_ import if_connected
import random

def shortest_path(G, source, target):
    shortest_path = nx.shortest_path_length(G, source, target, weight='weight')
    return shortest_path

def shortest_datasets_generation(config):
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
        while 1:
            random_graph = create_random_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            nodes = list(random_graph.nodes())
            random_nodes = random.sample(nodes, 2)
            if if_connected(random_graph, random_nodes[0], random_nodes[1]):
                path_value = shortest_path(random_graph, random_nodes[0], random_nodes[1])
            else:
                continue
            node_nums, edges_flat = graph_details_with_weight(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat, random_nodes[0], random_nodes[1])
            ans = config["answer"].format(path_value)
            # length check
            if len(tokenizer.encode(input_prompt + ans)) > 3000:
                continue
            sample = {}
            sample["input_prompt"] = input_prompt
            sample["answer"] = ans
            write_to_file(config["store_path"], sample)
            valid += 1
            if valid == config["samples_needed"][i]:
                break
            