# Reduce & message fuc
Build-in fuc: [dgl.function — DGL 0.8.0post2 documentation](https://docs.dgl.ai/api/python/dgl.function.html)

User define fuc: [User-defined Functions — DGL 0.8.0post2
documentation](https://docs.dgl.ai/api/python/udf.html)

从node端查看received message：dgl.udf.NodeBatch.mailbox

[dgl.udf.NodeBatch.mailbox— DGL 0.8.0post2 documentation](https://docs.dgl.ai/generated/dgl.udf.NodeBatch.mailbox.html#dgl.udf.NodeBatch.mailbox)

```python
# Definea UDF that computes the sum of the messages received and the original feature 
#for each node 
def node_udf(nodes):  
    # nodes.data['h'] is a tensor of shape(N, 1),  
    # nodes.mailbox['m'] is a tensor of shape(N, D, 1),  
    # where N is the number of nodes in the batch,  
    # D is the number of messages received per node for this node batch   
    return {'h': nodes.data['h'] + nodes.mailbox['m'].sum(1)}

g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 1, 0])))
g.ndata['h'] = torch.ones(2, 1)
g.update_all(fn.copy_u('h', 'm'), node_udf)
g.ndata['h']
```
