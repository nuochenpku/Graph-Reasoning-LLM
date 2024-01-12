import yaml
import argparse
import json
from transformers import LlamaTokenizer
import os
import networkx as nx
from itertools import combinations

def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)

def build_args():
    parser = argparse.ArgumentParser(description='GraphReasoner')
    parser.add_argument('--task', type=str, default='cycle_train', help='task to perform')
    return parser.parse_args()

def write_to_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str)
        f_converted.write('\n')

def graph_details(G):
    num_nodes = G.number_of_nodes()
    edges = G.edges()
    if isinstance(G, nx.DiGraph):
        edges_flat = " ".join(f"({i}->{j})" for i, j in edges)
    else:
        edges_flat = " ".join(str(edge) for edge in edges)
    return num_nodes, edges_flat

def graph_details_with_weight(G):
    num_nodes = G.number_of_nodes()
    edges_with_weights = [(u, v, d.get('weight', 1)) for u, v, d in G.edges(data=True)]
    if isinstance(G, nx.DiGraph):
        edges_flat = " ".join(f"({u}->{v},{w})" for u, v, w in edges_with_weights)
    else:
        edges_flat = " ".join(f"({u},{v},{w})" for u, v, w in edges_with_weights)
    return num_nodes, edges_flat

def graph_details_with_nodes_weight(G):
    num_nodes = G.number_of_nodes()
    nodes_info = [[node, G.nodes[node]['weight']] for node in G.nodes()]
    nodes_flat = " ".join(str(info) for info in nodes_info) 
    edges = G.edges()
    edges_flat = " ".join(str(edge) for edge in edges)
    return num_nodes, nodes_flat, edges_flat

def getTokenizer():
    return LlamaTokenizer.from_pretrained("/data/Llama-2-7b-hf")

def getMaxEdges(num_nodes):
    # add numbers of nodes and edges
    all_possible_edges = list(combinations(range(int(num_nodes)), 2))
    max_edges = len(all_possible_edges)
    return max_edges