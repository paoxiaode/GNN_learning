# Graph Matching

## Graph Matching Application

* Graph similar searching in graph based database
* 3D Action Recognition
* Unknown malware(恶意代码) detection 

## Problem Formulation

There are two kinds of graph matching formulation

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

Input: a pair of graph inputs $(G^1,G^2)$

* $G^1=(V^1,E^1)$ with $(X^1,A^1)$, where $X^1\in R^{N*d},A^1\in R^{N*N}$
* $G^2=(V^2,E^2)$ with $(X^2,A^2)$, where $X^2\in R^{N*d},A^2\in R^{N*N}$

Output

* $Y=\{-1,1\}$: graph-graph classification task
* $Y=[0,1]$: graph-graph regression task

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

* Use multiple stages of spectral clustering to create a multi-scale view of the similarity between graphs
* Align the nodes in the two graphs using the earth mover distance and computes correlation matrix in the aligned order.

### Graph-Bert

GB-DISTANCE (Zhang, 2020)

![image-20221012105326843](assets/image-20221012105326843.png)

Motivation: learn the distance metric of the graph instances, it is the foundation of many other research tasks, e.g., molecular graph clustering, brain graph classification and frequent sub-graph extraction

The common disadvantage 

* High computational cost: train cost will grow quadratically as graph number increases. the pair is $\frac{n(n-1)}{2}$
* Node-Order Invariant representation
  *  learned graph instance representations and the distance metric scores will vary greatly as the input graph node order changes

### Cross-graph  Matching

Graph Matching Networks (GMN) (Li et al, 2020)