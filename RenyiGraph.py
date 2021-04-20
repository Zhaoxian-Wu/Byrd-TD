# %%
import numpy as np
import random
#import matplotlib.pyplot as plt
#import networkx as nx

def Renyi(N, connectedRate, byzantineRate):
    edges = list()
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < connectedRate:
                edges.append((i, j))
    
    is_Byzantine_node = [random.random() < byzantineRate for _ in range(N)]
    honest_index = [i for i in range(N) if not is_Byzantine_node[i]]
    Byzantine_index = [i for i in range(N) if is_Byzantine_node[i]]
    
    edges = addSelfEdge(N, edges)
    connect = Connectivity(N, edges)
    return edges, honest_index, Byzantine_index, connect

def addSelfEdge(N, edges):
    return edges + [(i, i) for i in range(N)]

def Connectivity(N, edges):
    connect = np.zeros((N,N))
    for edge in edges:
        connect[edge[0],edge[1]] = 1
        connect[edge[1],edge[0]] = 1
        
    return connect
## 可视化
#def drawGraph(honestNodes, ByzantineNodes, edges, LostNodes=[], showLable=False):
#    NODE_COLOR_HONEST = '#99CCCC'
#    NODE_COLOR_BYZANTINE = '#FF6666'
#    NODE_COLOR_LOST = '#CCCCCC'
#    NODE_SIZE = 400
#    FONT_SIZE = 12
#    EDGE_WIDTH = 2
#
#    honestNodeSize = len(honestNodes)
#    ByzantineNodeSize = len(ByzantineNodes)
#    G = nx.empty_graph(honestNodeSize + ByzantineNodeSize)
#    
#    G.add_edges_from(edges)
#
#    # 布局
#    pos = nx.kamada_kawai_layout(G)
#
#    # honest agents
#    nx.draw_networkx_nodes(G, pos, 
#        node_size = NODE_SIZE, 
#        nodelist = honestNodes,
#        node_color = NODE_COLOR_HONEST,
#    )
#    # Byzantine agents
#    nx.draw_networkx_nodes(G, pos, 
#        node_size = NODE_SIZE,
#        nodelist = ByzantineNodes,
#        node_color = NODE_COLOR_BYZANTINE,
#    )
#    # Lost agents
#    nx.draw_networkx_nodes(G, pos, 
#        node_size = NODE_SIZE,
#        nodelist = LostNodes,
#        node_color = NODE_COLOR_LOST,
#    )
#
#    nx.draw_networkx_edges(G, pos, alpha=0.5, width=EDGE_WIDTH)
#
#    if showLable:
#        labels = {
#            i: str(i) for i in range(G.number_of_nodes())
#        }
#        nx.draw_networkx_labels(G, pos, labels, font_size=FONT_SIZE)

# 开始生成图
#edges, honest_index, Byzantine_index = Renyi(N=10, connectedRate=0.7, byzantineRate=0.2)
#
#for e in edges:
#    assert type(e) == tuple
#    assert len(e) == 2
#
## 图的信息
#N = len(honest_index) + len(Byzantine_index)
#is_Byzantine_node = [i in Byzantine_index for i in range(N)]
#
#neighbors = [
#    [v for v in range(N) if (n, v) in edges or (v, n) in edges]
#    for n in range(N)
#]
#honest_neighbors = [
#    [v for v in neighbors[n] if not is_Byzantine_node[v]]
#    for n in range(N)
#]
#Byzantine_neighbors = [
#    [v for v in neighbors[n] if is_Byzantine_node[v]]
#    for n in range(N)
#]
#
#honest_neighbors_size = [len(honest_neighbors[n]) for n in range(N)]
#Byzantine_neighbors_size = [len(Byzantine_neighbors[n]) for n in range(N)]
#
#LostNode = [
#    n for n in range(N) if not is_Byzantine_node[n]
#        and honest_neighbors_size[n] <= 2 * Byzantine_neighbors_size[n]
#]

## 可视化
#drawGraph(honest_index, Byzantine_index, edges, LostNode, showLable=True)