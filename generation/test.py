# from transformers import LlamaTokenizer

# t = "sfwefqewvewgw ewfwefg wegwe"

# tokenizer = LlamaTokenizer.from_pretrained("/data/Llama-2-7b-hf")

# tokens = tokenizer.encode(t)

# print(tokens)
# print(tokenizer.eos_token)

# import networkx as nx
# from networkx.algorithms import isomorphism

# def check_subgraph_isomorphism(G, g):
#     def is_match(G, g, mapping):
#         # 如果 g 中的所有节点都已经被映射，则检查是否为同构
#         if len(mapping) == len(g):
#             return all((mapping[u], mapping[v]) in G.edges() for u, v in g.edges())

#         # 选择还未被映射的一个节点
#         for node in g.nodes():
#             if node not in mapping:
#                 break

#         # 尝试在 G 中找到一个节点，它可以映射到 g 中的节点
#         for candidate in G.nodes():
#             if candidate not in mapping.values():
#                 # 尝试这个映射
#                 mapping[node] = candidate
#                 if is_match(G, g, mapping):
#                     return True
#                 # 回溯
#                 del mapping[node]

#         return False

#     return is_match(G, g, {})


# # 定义主图 G
# G = nx.Graph()
# G.add_edges_from([(0, 4), (0, 2), (0, 1), (0, 3), (1, 3), (2, 4), (2, 3), (3, 4)])

# # 定义子图 g
# g = nx.Graph()
# g.add_edges_from([(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)])

# if check_subgraph_isomorphism(G, g):
#     print("Yes")
# else:
#     print("No")


import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

G = nx.DiGraph()
G.add_edge(0, 3)
G.add_edge(0, 2)
G.add_edge(1, 3)
G.add_edge(1, 2)
G.add_edge(2, 3)

g = nx.DiGraph()
g.add_edge("a", "b")
# g.add_edge("a", "c")
# g.add_edge("c", "d")

matcher = DiGraphMatcher(G, g, node_match=None, edge_match=None)

is_isomorphic = matcher.subgraph_is_isomorphic()

print(is_isomorphic)