## GNN sample paradigms

![image-20221025213910084](./assets/image-20221025213910084.png)

### Node-wise Sample

GraphSage

![image-20221025213920472](./assets/image-20221025213920472.png)
$$
h_{\mathcal{N}(v)}^k\leftarrow AGGREGATE_k({h_u^{k-1}, \forall u \in \mathcal{N}(v)})\\
h_v^k \leftarrow \sigma (W^k \cdot  COMBINE(h_v^{k-1}, h_{\mathcal{N}(v)}^k))
$$


![image-20221025213929899](./assets/image-20221025213929899.png)

VR-GCN

通过历史嵌入向量来减少每一层需要采样的节点数

![image-20221025213940002](./assets/image-20221025213940002.png)

![image-20221025213948769](./assets/image-20221025213948769.png)

缺点：需要额外的内存来存储所有的历史隐藏嵌入，难以实现大规模扩展。



### Layer-wise Sample

引入不同节点的采样权重，在每层采样固定数量的节点

FastGCN

提出了基于重要度的采样模式，从而降低方差。在采样过程中，每一层的采样都是相互独立的，而且每一层的节点采样概率也保持一致。

![image-20221025215221303](./assets/image-20221025215221303.png)

ASGCN

Only samples nodes from the neighbors of the sampled node (yellow node) to obtain the better between-layer correlations, while FastGCN utilizes the importance sampling among all the nodes

![image-20221025220118598](./assets/image-20221025220118598.png)

![image-20221025220252885](./assets/image-20221025220252885.png)

### Graph-wise Sample

cluster-GCN

首先使用图分割算法将图分解为更小的子图，然后在子图层面上组成随机的分批，再将其输入 GNN 模型，从而降单次计算需求。

![image-20221025220434528](./assets/image-20221025220434528.png)

对于cluster-GCN，它只采样子图内的节点

![image-20221025220517951](./assets/image-20221025220517951.png)

Two problem：

* The links between sub-graphs are dismissed, which may fail to capture important correlations. 
* Clustering algorithm may change the original distribution of the dataset and introduce some bias.
* Solution: randomly combine some clusters(include the edges between clusters)

![image-20221025221054748](./assets/image-20221025221054748.png)

### Large-scale Graph Neural Networks on Recommendation Systems

**Application**

* item-item recommendation 
* user-item recommendation

#### item-item recommendation 

PinSage (第一次将GCN应用于工业级推荐系统的图算法,类似于GraphSage)

![image-20221025221756647](./assets/image-20221025221756647.png)

计算节点特征：从邻居节点聚合

* 从邻居节点经过dense层生成$n_u$
* 聚合$z_u,n_u$生成新的节点特征

![img](./assets/540b053872cb5bbc3cfed2410f31b477ecc5a747.jpg@942w_405h_progressive.webp)

我们希望负样本嵌入间的内积小于正样本嵌入间的内积，并且差值尽量大于某一阈值。对于节点$z_q, z_i$之间的loss，定义为

![img](./assets/8554395aad8226dac8dac92ab0cc9c7535d0386d.jpg@942w_80h_progressive.webp)

其中$n_k$为负样本，$\Delta $ 为超参数即阈值

#### user-item recommendation

IntentGC

![image-20221025224059491](./assets/image-20221025224059491.png)

![img](./assets/v2-01e86f2bc377041ed81a2ce78c5683d7_720w.webp)
