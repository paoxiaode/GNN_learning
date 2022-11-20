# Line Graph Neural Network

**Node classification or community detection ？**

community detection similar to K-means

Comparing to node classification, community detection **focuses on retrieving cluster information in the graph, rather than assigning a specific label to a node**. For example, as long as a node is clustered with its community members, it doesn’t matter whether the node is assigned as “community A”, or “community B”, while assigning all “great movies” to label “bad movies” will be a disaster in a movie network classification task.

## Community detection in a supervised setting

In this supervised setting, the model naturally predicts a label for each community. However, community assignment should be equivariant to label permutations. To achieve this, in each forward process, we take the minimum among losses calculated from all possible permutations of labels.

Mathematically, this means $L_{equivariant} = \underset{\pi \in S_c} {min}-\log(\hat{\pi}, \pi)$ where $S_c$ is the set of all permutations of labels, and $\hat{\pi}$ is the set of predicted labels

For instance, for a sample graph with node $\{1,2,3,4\}$ and community assignment $\{A,A,A,B\}$, with each node’s label $l∈\{0,1\}$,The group of all possible permutations$S_c = \{\{0,0,0,1\}, \{1,1,1,0\}\}$.