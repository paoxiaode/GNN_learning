# DGL paper

[[1909.01315v2] Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks (arxiv.org)](https://arxiv.org/abs/1909.01315v2)

contributions:

* DGL distills the computational patterns of GNNs into a few user-configurable message-passing
  primitives;
* DGL makes graph the central programming abstraction
* minimizing the effort it takes to port a model across frameworks.

## Message passing paradigm([Gilmer et al](https://arxiv.org/abs/1704.01212))

$$
m_{u\to v}^{(l)} = M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\to v}^{(l-1)}\right)\\
m_{v}^{(l)} = \sum_{u\in\mathcal{N}(v)}m_{u\to v}^{(l)}\\
h_v^{(l)} = U^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)
$$

$M^{(l)}$: the message func, $\sum$: the reduce func, $U^{(l)}$: the update func

## SPMM SDDMM

$$
\begin{aligned}
SPMM:& edge \rightarrow node (dgl.update\_all)\\
SDDMM:& node \rightarrow edge (dgl.apply\_ edge)
\end{aligned}
$$

SDDMM is that it maps the representation of an edge's incident nodes to the representation on the edge.

Similarly, SpMM aggregates the representation of a node's inbound edges into a node representation

### G-SPMM&G-SDDMM on tensorized hardware

workload require

* high computation to memory ratio: amortize the cost of memory op over many math op
* sufficient parallelism workload

optimize method

* preferred data format
* specific computation pattern
* fine-grained sync to ensure atomicity
