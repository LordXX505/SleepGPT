import networkx as nx
import matplotlib.pyplot as plt
import torch
from main.config import ex
import os
@ex.automain
def main(_config):
    plt.rcParams.update({
        'figure.figsize':(8,6)
    })
    G = nx.Graph()
    Node = ['C3', 'C4', 'F3', 'F4', 'Fpz', 'O1', 'O2', 'Pz']
    index = torch.tensor([0, 1, 5, 6, 7, 8, 9, 10])
    for node in Node:
        G.add_node(node)
    ckpt = torch.load(_config["load_path"], map_location="cpu")
    print(_config["load_path"])
    index = index.unsqueeze(-1).repeat(1, ckpt['state_dict']['transformer.pos_embed'][0].shape[1])
    row = torch.gather(ckpt['state_dict']['transformer.pos_embed'][0].detach().clone(), dim=0, index=index)
    col = torch.gather(ckpt['state_dict']['transformer.pos_embed'][0].detach().clone(), dim=0, index=index)
    distance = row.unsqueeze(0) - col.unsqueeze(1)
    distance = torch.sum(distance, dim=-1)
    n = len(distance)
    for i in range(n):
        for j in range(n):
            G.add_weighted_edges_from([(Node[i], Node[j], distance[i][j])])
    print(G.graph)
    nx.draw(G, with_labels=True)
    plt.savefig('/home/hwx/Sleep/result/graph.png')
    # path = '/'.join(_config['load_path'].split('/')[-4:-2])
    # print(f'../../result/{path}')
    # os.makedirs(f'../../result/{path}', exist_ok=True)
    # plt.savefig(f'../../result/{path}/mask{id}.png')
    # plt.close("all")n
