# Link prediction in GNN

Application:

* social network friend recommendation
* co-authorship prediction in citation network
* movie recommendation in Netflix
* protein interaction prediction in biological networks



Two popular GNN-based link prediction paradigms

* Node-based
  * Learn node representation through a GNN 
  * aggregate the pairwise node representations as link representations
* Subgraph-based
  * extract a local subgraph around each target link, and then apply a graph-level GNN (with pooling) to each subgraph to learn a subgraph representation



## Traditional methods

### Local heuristics

![image-20221031203142566](./assets/image-20221031203142566.png)

$\Gamma(x)$ to denote the set of $x$’s neighbors

![image-20221031202620949](./assets/image-20221031202620949.png)

考虑节点间相同邻居的数量

![image-20221031202628844](./assets/image-20221031202628844.png)



![image-20221031202848384](./assets/image-20221031202848384.png)

![image-20221031203047046](./assets/image-20221031203047046.png)

节点的度越高，权重越低，两个节点的相连可能性低

### Global Heuristics

![image-20221031203349961](./assets/image-20221031203349961.png)

计算两节点间不同长度路径的数量之和，长度越长权重越小



## GNN-based

### GAE: Graph-AutoEncoder

自监督的训练，训练网络来预测图中原有的边，进而预测

Given the adjacency matrix $A$ and node feature matrix $X$

For each i,j , output a probability for link $(i,j)$

![image-20221031203745758](./assets/image-20221031203745758.png)

consider $A_{i,j}=\{0,1\}$

![image-20221031204049455](./assets/image-20221031204049455.png)

### VGAE: Variational Graph auto-encoder

[VGAE（Variational graph auto-encoders）论文详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/78340397)

[VGAE：图变分自编码器 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/345337636)

![(Variational) Graph Auto-Encoder](./assets/figure.png)

#### 编码器

定义了一个后验概率

![image-20221031214531490](./assets/image-20221031214531490.png)

编码器为两层图卷积网络，得到每个节点的向量$z_i$

![image-20221031214538953](./assets/image-20221031214538953.png)

#### 解码器

解码器计算图中任意两点之间存在边的概率来重构图：

![image-20221031214905229](./assets/image-20221031214905229.png)

损失函数包括似然期望（越大），以及节点表示向量分布和正态分布的KL散度（越小）两部分：

[Variational Autoencoder（变分自编码 VAE，CVAE）](https://blog.csdn.net/qq_39388410/article/details/79129197)

![image-20221031221005696](./assets/image-20221031221005696.png)

### 代码细节

**examples/pytorch/vgae/train.py**

```python
def encoder(self, g, features):
        h = self.layers[0](g, features)
        self.mean = self.layers[1](g, h)
        self.log_std = self.layers[2](g, h)
        gaussian_noise = torch.randn(features.size(0), self.hidden2_dim).to(device)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(device)
        return sampled_z

def decoder(self, z):
    adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
    return adj_rec

def forward(self, g, features):
    z = self.encoder(g, features)
    adj_rec = self.decoder(z)
    return adj_rec
```

![image-20221101210508712](./assets/image-20221101210508712.png)



### Variants of GAE and VGAE

* replace the GCN used in GAE/VGAE with any GNN
* replace the inner product with any aggregation function over $\{z_i , z_j\}$ and feed the aggregated link representation to an MLP to predict the link $(i, j)$.

HGCN combines hyperbolic graph convolutional neural networks with a Fermi-Dirac decoder for aggregating pairwise node embeddings and outputting link probabilities

![image-20221101211559122](./assets/image-20221101211559122.png)

## Subgraph-Based

### SEAL

![image-20221102094711917](./assets/image-20221102094711917.png)

1. **Enclosing subgraph**

   Given a set of nodes $S$
   $$
   \cup_{j\in S}\{i|d(i,j)\leq h\}
   $$
   the $h$-hop enclosing subgraph around a node set $S$ contains nodes within $h$ hops of any node in $S$

2. **Node labeling**

   SEAL applies a Double Radius Node Labeling (DRNL) to give an integer label to each node in the subgraph as its additional feature

![image-20221102100927816](./assets/image-20221102100927816.png)
