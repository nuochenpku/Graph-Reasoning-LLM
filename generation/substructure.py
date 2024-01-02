from networkx.algorithms import isomorphism

def count_subgraph_occurrences(G, g):
    matcher = isomorphism.DiGraphMatcher(G, g)
    match_count = sum(1 for _ in matcher.subgraph_isomorphisms_iter())
    return match_count