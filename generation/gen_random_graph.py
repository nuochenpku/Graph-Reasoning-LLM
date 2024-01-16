import networkx as nx
from random import randint, sample, shuffle
from itertools import combinations, permutations
from itertools import product

def create_random_graph(min_nodes, max_nodes, max_edges, minimum, maximum, weight=False, directed=True):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = randint(min_nodes, max_nodes)
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(num_nodes)) 
    
    max_possible_edges = num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(minimum * max_possible_edges)
    upper_limit_edges = int(maximum * max_possible_edges)

    num_edges_to_add = randint(lower_limit_edges, min(upper_limit_edges, max_edges))

    if num_edges_to_add == 0:
        num_edges_to_add = 1
    
    all_possible_edges = list(combinations(range(num_nodes), 2))
    edges_to_add = sample(all_possible_edges, num_edges_to_add) if num_edges_to_add > 0 else []
    
    if weight:
        edges_with_weights = [(u, v, randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges_with_weights = [(u, v, 0) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    
    return G


import networkx as nx
import random
from itertools import product

def create_random_bipartite_graph(min_nodes, max_nodes, max_edges, minimum, maximum, weight=False, directed=False):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")

    num_nodes = random.randint(min_nodes, max_nodes)
    
    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    split_index = random.randint(1, num_nodes - 1)
    nodes_set_1 = nodes[:split_index]
    nodes_set_2 = nodes[split_index:]
    
    max_possible_edges = len(nodes_set_1) * len(nodes_set_2)

    lower_limit_edges = int(minimum * max_possible_edges)
    upper_limit_edges = int(maximum * max_possible_edges)
    
    num_edges_to_add = random.randint(lower_limit_edges, min(upper_limit_edges, max_edges))
    if num_edges_to_add == 0:
        num_edges_to_add = 1

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)

    all_possible_edges = list(product(nodes_set_1, nodes_set_2))
    random.shuffle(all_possible_edges)  

    edges_to_add = all_possible_edges[:num_edges_to_add]
    if weight:
        edges_with_weights = [(u, v, random.randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        G.add_edges_from(edges_to_add)

    return G


def create_smaller_graph(min_nodes, max_nodes, max_edges, minimum, maximum, weight=False, directed=True):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = max(randint(min_nodes, max_nodes)/2, 3)
    # num_nodes = min(num_nodes, 10)
    
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(range(int(num_nodes)))     
    max_possible_edges = num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(minimum * max_possible_edges)
    upper_limit_edges = int(maximum * max_possible_edges)
    
    num_edges_to_add = randint(lower_limit_edges, min(upper_limit_edges, max_edges))/2

    if num_edges_to_add == 0:
        num_edges_to_add = 1
    
    all_possible_edges = list(combinations(range(int(num_nodes)), 2))
    edges_to_add = sample(all_possible_edges, int(num_edges_to_add)) if num_edges_to_add > 0 else []
    
    if weight:
        edges_with_weights = [(u, v, randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges_with_weights = [(u, v, 0) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    
    return G

def create_random_graph_node_weights(min_nodes, max_nodes, max_edges, minimum, maximum, directed=True):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")

    num_nodes = randint(min_nodes, max_nodes)

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes with random weights
    for node in range(num_nodes):
        G.add_node(node, weight=randint(1, 10))

    max_possible_edges = num_nodes * (num_nodes - 1) if directed else num_nodes * (num_nodes - 1) // 2

    lower_limit_edges = int(minimum * max_possible_edges)
    upper_limit_edges = int(maximum * max_possible_edges)

    num_edges_to_add = randint(lower_limit_edges, min(upper_limit_edges, max_edges))

    if num_edges_to_add == 0:
        num_edges_to_add = 1

    all_possible_edges = list(permutations(range(num_nodes), 2)) if directed else list(combinations(range(num_nodes), 2))
    edges_to_add = sample(all_possible_edges, num_edges_to_add) if num_edges_to_add > 0 else []

    G.add_edges_from(edges_to_add)

    return G


def create_topology_graph(min_nodes, max_nodes, minimum, maximum, weight=False, directed=True):
    if not directed:
        raise ValueError("The graph must be directed to ensure a unique topological path.")
    
    num_nodes = randint(min_nodes, max_nodes)
    G = nx.DiGraph()
    
    nodes = list(range(num_nodes))
    shuffle(nodes)
    G.add_nodes_from(nodes)
    
    for i in range(num_nodes - 1):
        G.add_edge(nodes[i], nodes[i + 1], weight=randint(1, 10) if weight else 0)
    
    max_possible_additional_edges = num_nodes * (num_nodes - 1) - (num_nodes - 1)
    
    target_min_edges = int(minimum * max_possible_additional_edges)
    target_max_edges = int(maximum * max_possible_additional_edges)
    
    target_max_edges = min(target_max_edges, max_possible_additional_edges)
    
    num_additional_edges = randint(target_min_edges, target_max_edges)
    
    possible_additional_edges = [(u, v) for u, v in permutations(nodes, 2) if not G.has_edge(u, v)]
    
    additional_edges = sample(possible_additional_edges, num_additional_edges)
    
    for u, v in additional_edges:
        if not nx.has_path(G, v, u): 
            G.add_edge(u, v, weight=randint(1, 10) if weight else 0)
    
    return G
