from networkx.algorithms import isomorphism
from utils import write_to_file, getTokenizer, graph_details
from gen_random_graph import create_random_graph, create_smaller_graph
import networkx as nx
import string

# def count_subgraph_occurrences(G, g):
#     matcher = isomorphism.DiGraphMatcher(G, g)
#     match_count = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
#     return match_count

def count_subgraph_occurrences(G, g):
    # Prune candidate nodes based on degree and other checks
    def can_map(u, v):
        # Add additional checks here if needed
        return G.degree(v) >= g.degree(u)

    # Improved backtracking using vertex invariants
    def is_match(mapping, g_nodes, G_nodes):
        if len(mapping) == len(g_nodes):
            return 1  # Return count instead of boolean
        else:
            total_count = 0
            u = next(node for node in g_nodes if node not in mapping)
            for v in G_nodes:
                if v not in mapping.values() and can_map(u, v):
                    mapping[u] = v
                    total_count += is_match(mapping, g_nodes, G_nodes)
                    del mapping[u]
            return total_count

    g_nodes_sorted = sorted(g.nodes(), key=lambda x: (-g.degree(x), x))
    G_nodes_sorted = sorted(G.nodes(), key=lambda x: (-G.degree(x), x))
    
    return is_match({}, g_nodes_sorted, G_nodes_sorted)

def relabel_nodes_to_alphabet(G):
    mapping = dict(zip(G.nodes(), string.ascii_lowercase))
    H = nx.relabel_nodes(G, mapping)
    return H

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
            smaller_graph = relabel_nodes_to_alphabet(smaller_graph)
            node_nums_2, edges_flat_2 = graph_details(smaller_graph)
            input_prompt = config["prompt"].format(0, node_nums_1-1, edges_flat_1, 'a', string.ascii_lowercase[node_nums_2-1], edges_flat_2)
            # size check, judge if smaller_graph has smaller size than random_graph
            print(smaller_graph.number_of_edges(), random_graph.number_of_edges())
            if node_nums_2 > node_nums_1 or smaller_graph.number_of_edges() > random_graph.number_of_edges():
                continue
            # duplicate check
            if input_prompt in dup:
                continue
            dup.add(input_prompt)
            # connectivity check
            if not nx.is_weakly_connected(smaller_graph):
                continue
            # if match_count == 0 or match_count > 100:
            #     continue
            if match_count != 0:
                continue
            if match_count:
                ans = config["answer"].format("Yes")
            else:
                ans = config["answer"].format("No")
            # ans = config["answer"].format(match_count)
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