import json
import networkx as nx

def parse_graph(edges_str, directed=True):
    edges = edges_str.strip().split(') ')
    graph = nx.DiGraph() if directed else nx.Graph()
    for edge in edges:
        if edge:  # Check if edge is not empty
            edge = edge.strip('()')
            nodes = edge.split('->')
            if len(nodes) == 2:  # Ensure we have exactly two nodes
                graph.add_edge(nodes[0].strip(), nodes[1].strip())
            else:
                raise ValueError(f"Edge parsing failed for edge: {edge}")
    return graph

def check_subgraph_isomorphism(G, g):
    GM = nx.algorithms.isomorphism.DiGraphMatcher(G, g)
    if GM.subgraph_is_monomorphic():
        for iso in GM.subgraph_monomorphisms_iter():
            print({g_node: G_node for G_node, g_node in iso.items()})
        return True
    return False

def test_subgraph_isomorphism(file_path):
    s = 0
    a = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            s += 1
            data = json.loads(line)
            input_prompt = data['input_prompt']
            answer = data['answer'].split('###')[-1].strip()

            # Extract graph G and subgraph G' edges from the input prompt
            G_edges_str = input_prompt.split("edges are:")[1].split(". The nodes of subgraph G'")[0]
            g_edges_str = input_prompt.split("edges are:")[2].split(". Is")[0]

            # Create graph G and subgraph G'
            G = parse_graph(G_edges_str)
            g = parse_graph(g_edges_str)

            # Check subgraph isomorphism and compare with the provided answer
            result = 'Yes' if check_subgraph_isomorphism(G, g) else 'No'

            # assert result == answer, f"Test failed: Expected answer was {answer}, but got {result}."
            if result == answer:
                a += 1
            
            print(f"Test {s}:", a/s, f"({a}/{s})")

            print(f"Test passed: Expected answer was {answer}, and got {result}.")

# The path to the JSON file
file_path = '/home/yuhanli/Graph-Reasoning-LLM/datasets/train_set/substructure_train.json'
test_subgraph_isomorphism(file_path)