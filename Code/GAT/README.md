# GAT

Environment:

python 3.9

dgl 0.8

torch==1.10.1+cu113


full graph training

```
python main.py
```

mini-batch training

```
python train_sample.py
```


[Understand Graph
Attention Network — DGL 0.8.0post2 documentation](https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html)

The
difference between GCN and GAT:

How the information from the one-hop neighborhood
is aggregated

GCN:

$$
h_i^{(l+1)}=\sigma\left(\sum_{j\in \mathcal{N}(i)} {\frac{1}{c_{ij}} W^{(l)}h^{(l)}_j}\right)
$$

* $N(i)$: the set of one-hop neighbors
* $c_{ij}=\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}$:normalization constant![]()
* $W^{(l)}$: shared wight matrix for node-wise feature transformation

GAT:不同于GraphSage直接把邻居feature相加后平均，GAT引入注意力系数将不同权重给不同邻居的feature
<center>
<img src=image/README/gat.png style="zoom:50%">
</center>

$$
\begin{aligned}
z_i^{(l)}&=W^{(l)}h_i^{(l)},&(1) \\
e_{ij}^{(l)}&=\text{LeakyReLU}(\vec a^{(l)^T}(z_i^{(l)}||z_j^{(l)})),&(2)\\
\alpha_{ij}^{(l)}&=\frac{\exp(e_{ij}^{(l)})}{\sum_{k\in \mathcal{N}(i)}^{}\exp(e_{ik}^{(l)})},&(3)\\
h_i^{(l+1)}&=\sigma\left(\sum_{j\in \mathcal{N}(i)} {\alpha^{(l)}_{ij} z^{(l)}_j }\right),&(4)
\end{aligned}
$$

* Equation(1): linear transformation, learnabweight matrix $W^{(l)}$
* Equation(2): compute a pair-wise  un-normalized attention score between two neighbors
  * First, concatenate the embeddings of two nodes
  * Then take dot-product of it and a learnable weight vector $a^{(l)}$
  * Apply a LeakyRelu in the end
* Equation (3) applies a softmax to normalize the attention scores on each node’s incoming edges.
* Equation (4) is similar to GCN. The embeddings from neighbors are aggregated together, scaled by the attention scores.

# Multi-head attention

Motivation: enrich the model capacity and to
stabilize the learning process

Each head have its own parameters and outputs, two ways to merge outputs:

$$\text{concatenation}: h^{(l+1)}_{i} =||_{k=1}^{K}\sigma\left(\sum_{j\in \mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)
$$
or
$$
\text{average}: h_{i}^{(l+1)}=\sigma\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{k}W^{k}h^{(l)}_{j}\right)
$$

一般在中间隐藏层采用concatenation，最后输出层采用average
