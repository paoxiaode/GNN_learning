# GNN_learning

This is a repo about GNN and DGL

## GNN 入门

### 代码

* 安装wsl2（windows的linux子系统）
* 配python环境+安装DGL [Deep Graph Library (dgl.ai)](https://www.dgl.ai/)
* 读DGL里面三个模型GCN, RGCN, GAT的代码，跑一下example

### 论文

* 下载zotero
* [A Gentle Introduction to Graph Neural Networks (distill.pub)](https://distill.pub/2021/gnn-intro/)

* 综述论文：Graph Neural Networks: A Review of Methods and Applications
* 了解GNN的基本模型：GCN (Graph convolution network), RGCN (Relational graph convolution network), GAT (Graph attention network)

## reference:

### model
GCN (Graph convolutional network)

* [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)
* [Graph Convolutional Network — DGL 0.8.0post2 documentation](https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html)

GAT (Graph attention network)

* [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907)
* [Understand Graph Attention Network — DGL 0.8.0post2 documentation](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)

RGCN (Relational graph convolutional network)

* [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103.pdf)
* [Relational Graph Convolutional Network — DGL 0.8.0post2 documentation](https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html)

TGN (temporal graph network)

* [Accelerating and scaling Temporal Graph Networks on the Graphcore IPU](https://www.handla.it/accelerating-and-scaling-temporal-graph-networks-on-the-graphcore-ipu-by-michael-bronstein-jun-2022/)

* [twitter-research/tgn: TGN: Temporal Graph Networks (github.com)](https://github.com/twitter-research/tgn)

### DGL 

[build-in function in DGL](https://docs.dgl.ai/api/python/dgl.function.html)


## Notebooks:
[reduce and message function in DGL](Notes/reduce&message.md)
