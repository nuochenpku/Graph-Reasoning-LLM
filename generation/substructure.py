from networkx.algorithms import isomorphism
from utils import write_to_file, getTokenizer, graph_details
from gen_random_graph import create_random_graph, create_smaller_graph

def count_subgraph_occurrences(G, g):
    matcher = isomorphism.DiGraphMatcher(G, g)
    match_count = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
    return match_count

def substruc_datasets_generation(config):
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
            smaller_graph = create_smaller_graph(min_nodes, max_nodes, max_edges, min_ratio, max_ratio, weight, directed)
            match_count = count_subgraph_occurrences(random_graph, smaller_graph)
            node_nums_1, edges_flat_1 = graph_details(random_graph)
            node_nums_2, edges_flat_2 = graph_details(smaller_graph)
            input_prompt = config["prompt"].format(0, node_nums_1-1, edges_flat_1, 0, node_nums_2-1, edges_flat_2)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            if match_count == 0 or match_count > 100:
                continue
            ans = config["answer"].format(match_count)
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