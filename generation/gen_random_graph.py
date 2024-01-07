import networkx as nx
from random import randint, sample
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


def create_random_bipartite_graph(min_nodes, max_nodes, max_edges, minimum, maximum, weight=False, directed=False):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = max(randint(min_nodes, max_nodes), 2)
    
    middle_index = num_nodes // 2
    nodes_set_1 = range(0, middle_index)
    nodes_set_2 = range(middle_index, num_nodes)
    max_possible_edges = len(nodes_set_1) * len(nodes_set_2)
    
    lower_limit_edges = int(minimum * max_possible_edges)
    upper_limit_edges = int(maximum * max_possible_edges)
    
    num_edges_to_add = randint(lower_limit_edges, min(upper_limit_edges, max_edges))

    if num_edges_to_add == 0:
        num_edges_to_add = 1
    
    all_possible_edges = list(product(nodes_set_1, nodes_set_2))
    edges_to_add = sample(all_possible_edges, num_edges_to_add) if num_edges_to_add > 0 else []
    
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    if weight:
        edges_with_weights = [(u, v, randint(1, 10)) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    else:
        edges_with_weights = [(u, v, 0) for u, v in edges_to_add]
        G.add_weighted_edges_from(edges_with_weights)
    
    return G

def create_smaller_graph(min_nodes, max_nodes, max_edges, minimum, maximum, weight=False, directed=True):
    if not 0 <= maximum <= 1 or not 0 <= minimum <= 1:
        raise ValueError("Sparsity must be between 0 and 1.")
    
    num_nodes = max(randint(min_nodes, max_nodes)/4, 4)
    
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