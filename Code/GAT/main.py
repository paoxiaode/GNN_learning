import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl import DGLGraph
from dgl.data import CoraGraphDataset
import time
import sys
import numpy as np

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim) -> None:
        super(GATLayer, self).__init__()
        self.g =g
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.cal_att = nn.Linear(in_features=2*out_dim, out_features=1)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.cal_att.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.concat([edges.src["z"], edges.dst["z"]], dim = 1)
        a = self.cal_att(z2)
        return {'e': F.leaky_relu(a)}
    
    def message_func(self, edges):
        return {"z":edges.src["z"], 'e': edges.data['e']}
    
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim = 1)
        return {'h': h}
    
    def forward(self, h):
        z = self.linear(h)
        self.g.ndata["z"] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        
        return self.g.ndata.pop("h")
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.g = g
        for _ in range(num_heads):
            self.heads.append(GATLayer(self.g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs), dim=0)
        
class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, num_layer=2) -> None:
        super(GAT, self).__init__()
        self.g = g
        self.num_layer = num_layer
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadGATLayer(self.g, in_dim, hidden_dim, num_heads))
        for _ in range(1, self.num_layer):
            self.layers.append(MultiHeadGATLayer(self.g, hidden_dim*num_heads, hidden_dim, num_heads))
        self.layers.append(MultiHeadGATLayer(self.g, hidden_dim*num_heads, out_dim, num_heads, merge="mean"))
        
    def forward(self, h):
        # print(h.shape)
        for layer in self.layers[:-1]:
            h = layer(h)
            h = F.elu(h)
            # print(h.shape)

        h = self.layers[-1](h)
        # print(h.shape)

        return h
        
def run():
    dataset = CoraGraphDataset()
    graph = dataset[0]
    train_mask = graph.ndata["train_mask"]
    features = graph.ndata["feat"]
    labels = graph.ndata["label"]
    num_classes = dataset.num_classes
    num_edges = graph.num_edges()
    num_nodes = graph.num_nodes()
    in_dim = features.shape[1]
    print(f"""num edges {num_edges},
          num nodes {num_nodes}
          num classes {num_classes}
          input feature dim {in_dim}""")
    net = GAT(graph, in_dim, hidden_dim=8, out_dim= num_classes, num_heads=2, num_layer=3)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    dur = []
    
    for epoch in range(30):
        if epoch >= 3:
            t0 = time.time()
            
        logits = net(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # exit()
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur))) 
    
    
    
if __name__ == "__main__":
    run()