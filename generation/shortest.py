import networkx as nx
from utils import write_to_file, getTokenizer, graph_details_with_weight
from gen_random_graph import create_random_graph
from connectivity_ import if_connected
import random
from utils import getMaxEdges

def shortest_path(G, source, target):
    shortest_path = nx.shortest_path_length(G, source, target, weight='weight')
    return shortest_path

def shortest_datasets_generation(config):
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
        # edges_number = [int(getMaxEdges(min_nodes) * min_ratio), int(getMaxEdges(max_nodes) * max_ratio)]
        edges_number = [min_ratio, max_ratio]
        nodes_number = [min_nodes, max_nodes]
        valid = 0
        dup = set()
        # test duplicate check
        # if "test" in config["store_path"]:
        #     # read from train
        #     with open("../datasets/train_set/shortest_train.json", "r") as f:
        #         for line in f:
        #             if line.strip() == "":
        #                 continue
        #             sample = eval(line.strip())
        #             dup.add(sample["input_prompt"])
        while 1:
            random_graph = create_random_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            nodes = list(random_graph.nodes())
            random_nodes = random.sample(nodes, 2)
            # edge number check
            if random_graph.number_of_edges() == 1:
                continue
            # edge check
            if random_graph.has_edge(random_nodes[0], random_nodes[1]):
                continue
            if if_connected(random_graph, random_nodes[0], random_nodes[1]):
                path_value = shortest_path(random_graph, random_nodes[0], random_nodes[1])
            else:
                continue
            node_nums, edges_flat = graph_details_with_weight(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat, random_nodes[0], random_nodes[1])
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            ans = config["answer"].format(path_value)
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
            