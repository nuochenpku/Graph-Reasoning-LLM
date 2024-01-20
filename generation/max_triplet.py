import itertools
from utils import write_to_file, getTokenizer, graph_details_with_nodes_weight
from gen_random_graph import create_random_graph_node_weights
from cycle_ import if_cyclic
from utils import getMaxEdges

def maximum_triplet_sum(G):
    max_triplet_sum = 0
    for triplet in itertools.combinations(G.nodes(data=True), 3):
        (node1, data1), (node2, data2), (node3, data3) = triplet
        if G.has_edge(node1, node2) and G.has_edge(node2, node3) and G.has_edge(node3, node1):
            sum_triplet = data1['weight'] + data2['weight'] + data3['weight']
            max_triplet_sum = max(max_triplet_sum, sum_triplet)
    return max_triplet_sum

def triplet_datasets_generation(config):
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
        #     with open(config["store_path"].replace("test", "train"), "r") as f:
        #         for line in f:
        #             if line.strip() == "":
        #                 continue
        #             sample = eval(line.strip())
        #             dup.add(sample["input_prompt"])
        while 1:
            random_graph = create_random_graph_node_weights(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, directed)
            max_triplet = maximum_triplet_sum(random_graph)
            node_nums, nodes_flat, edges_flat = graph_details_with_nodes_weight(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, nodes_flat, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            ans = config["answer"].format(max_triplet)
            # length check
            if len(tokenizer.encode(input_prompt + ans)) > 3000:
                continue
            # triplet check
            if max_triplet == 0:
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