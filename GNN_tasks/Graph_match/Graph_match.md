# Graph Matching

## Graph Matching Application

* Graph similar searching in graph based database
* 3D Action Recognition
* Unknown malware(恶意代码) detection 

## Problem Formulation

Given a set of $m$ graph instances $\mathcal{G}=\{G^1,G^2,\dots,G^m\}$, we aim to learn a mapping $d:G*G\rightarrow R$, mapping $d(·, ·)$ should also maintain the basic distance metric properties, i.e., nonnegativity, identity of indiscernibles, symmetry and triangle inequality.

### Node Correspondence

![image-20221008101601871](./assets/image-20221008101601871.png)

Find a node-to-node correspondence matrix between two graphs.

* NP-hard
* Computationally expensive and Poor scalability

### Similarity Learning

![image-20221008101853639](./assets/image-20221008101853639.png)

Produce a similarity score between two graphs, usually use **GED (Graph edit distance) and MCS (Most common subgraph)**

<img src="assets/image-20221012100159991.png" alt="image-20221012100159991" style="zoom: 80%;" />

* NP-hard
* GNN-based methods demonstrating superiority over traditional methods

## Graph Similarity Learning methods

### Convolutional  Set Matching

GraphSim(Bai et al, 2020b): 

<img src="assets/image-20221012100433275.png" alt="image-20221012100433275" style="zoom:67%;" />

* 节点间的顺序是怎么确定的？BFS

* 在得到相似度矩阵后怎么操作？类似于CNN(卷积、池化啥的)

### Hierarchical clustering

![image-20221012095059123](./assets/image-20221012095059123.png)

Hierarchical graph matching network (HGMN) (Xiu et al, 2020)

Motivation:  two similar graphs should also be similar when they are compressed into more compact graphs.

**HGMN fundamental difference:**

* Use multiple stages of spectral clustering to create a multi-scale view of the similarity betw een graphs
* Align the nodes in the two graphs using the earth mover distance and computes correlation matrix in the aligned order.

### Graph-Bert

亮点：提出了一种计算graph-pair distance metric的方案，引入了semi-supervised，同时计算时考虑了metric的基本性质

GB-DISTANCE (Zhang, 2020)

![image-20221012105326843](assets/image-20221012105326843.png)

Motivation: learn the distance metric of the graph instances, it is the foundation of many other research tasks, e.g., molecular graph clustering, brain graph classification and frequent sub-graph extraction

#### The common disadvantage of graph match model

* High computational cost: train cost will grow quadratically as graph number increases. the pair is $\frac{n(n-1)}{2}$
* Node-Order Invariant representation
  *  learned graph instance representations and the distance metric scores will vary greatly as the input graph node order changes
  *  ![image-20221013162044870](assets/image-20221013162044870.png)
* Semi-supervised learning
  * The graph-pair distance scores need to be labeled, tedious and time-consuming.

* Lack of metric properties

#### Framework

##### Graph representation gen: Graph-Bert + fusion

For node $v_i$, initial input embedding is 

![image-20221018104421308](assets/image-20221018104421308.png)

Which pre-computed WL code as $WL(v_i)\in N$, $w_i=[w(v_i,v_j)]_{v_j\in v}\in R^{|V|*1}$, $D(v_i) ∈ N$ is the degree of $v_i$.
$$
e_i^x=Embed(x_i)\in R^{d_h*1}
$$


![image-20221018103954874](assets/image-20221018103954874.png)

![image-20221018104253542](assets/image-20221018104253542.png)



After init, we can get $H$: the all node representations

![image-20221018104926481](assets/image-20221018104926481.png)

**G-Transformer:**

![image-20221018105210365](assets/image-20221018105210365.png)

$z$ is the mean of all node representation in a graph 

![image-20221018105228361](assets/image-20221018105228361.png)

##### Distance metric and loss

The distance between two graphs:

![image-20221018110307750](assets/image-20221018110307750.png)

* non-negative
* normal range [0,1]
* d=0 for the same $z$

then we can get matrix $D, D(i,j)=d(G^{(i)},G^{(j)})$

![image-20221018110630709](assets/image-20221018110630709.png)

which $\alpha\in[0,1]$ is hyper-parameter, $\beta$ is a large number.

### Cross-graph  Matching

Graph Matching Networks (GMN) (Li et al, 2020)

Two models for graph similarity learning: **graph embedding/matching network**

<img src="assets/image-20221018112204533.png" alt="image-20221018112204533" style="zoom: 150%;" />

#### Graph embedding models (traditional GNN)

Generate node representation

<img src="assets/image-20221018112511213.png" alt="image-20221018112511213" style="zoom:130%;" />

Aggregation to graph representation

![image-20221018112645899](assets/image-20221018112645899.png)

Complexity: $O(|V|+|E|)$

#### GMN (Graph Matching Network)

Given two graphs $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ , produces the similarity score $s(G_1, G_2)$

<img src="assets/image-20221018113427601.png" alt="image-20221018113427601" style="zoom: 50%;" />

$f_{match}$ is a function that communicates cross graph information

![image-20221018114138343](assets/image-20221018114138343.png)

Complexity: $O(|V_1||V_2|)$

##### Loss

**Pairwise training**: requires a dataset of pairs labeled as positive (similar) or negative (dissimilar)

![image-20221018115638917](assets/image-20221018115638917.png)

where $t ∈ \{−1, 1\}$ is the label for this pair, $γ > 0$ is a margin parameter, and $d(G_1, G_2) = ‖h_{G_1} − h_{G_2} ‖^2$ is the Euclidean distance. This loss encourages $d(G_1, G_2) < 1−γ$ when the pair is similar $(t = 1)$, and $d(G_1, G_2) > 1+γ$ when $t = −1$.



**Triplet training**: 

![image-20221018115806818](assets/image-20221018115806818.png)