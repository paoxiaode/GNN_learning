import dgl
import torch

g = dgl.graph(([1, 2], [2, 3]))
block = dgl.to_block(g, torch.LongTensor([3, 2]))
g = dgl.graph(([1, 3], [2, 2]))

block = dgl.to_block(g, torch.LongTensor([2]))
