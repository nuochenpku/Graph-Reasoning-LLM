from utils import getTokenizer, write_to_file, graph_details
from gen_random_graph import create_random_graph

def hamiltonian_path(G):
    N = len(G.nodes())
    path = []
    
    def backtrack(node):
        path.append(node)
        if len(path) == N:
            return path
        for neighbor in G.neighbors(node):
            if neighbor not in path:
                result = backtrack(neighbor)
                if result is not None:
                    return result
        path.pop()
        return "No path found"
    
    for starting_node in G.nodes():
        path = backtrack(starting_node)
        if path is not None:
            return path
            
    return "No path found"

def hamiltonian_datasets_generation(config):
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
            path = hamiltonian_path(random_graph)
            node_nums, edges_flat = graph_details(random_graph)
            input_prompt = config["prompt"].format(0, node_nums-1, edges_flat)
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            if path == "No path found":
                ans = config["answer"].format("No")
            else:
                ans = config["answer"].format("Yes")
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
            