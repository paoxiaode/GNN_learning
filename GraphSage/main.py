from pickletools import optimize
import dgl
from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.data
import argparse

class SAGEConv(nn.Module):
    def __init__(self, in_feat, out_feat) -> None:
        super(SAGEConv, self).__init__()
        self.linear = nn.Linear(2*in_feat, out_feat)
        
    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['h'] = feat
            g.update_all(message_fuc = fn.copy_u("h", "m"), reduce_func = fn.mean("m", "h_N"))
            h_N = g.ndata['h_N']
            h_total = torch.concat([feat, h_N], dim=1)
            return self.linear(h_total)
        
        
class Model(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            self.layers.append(SAGEConv(in_feats, n_hidden))
            for i in range(1, n_layers - 1):
                self.layers.append(SAGEConv(n_hidden, n_hidden))
            self.layers.append(SAGEConv(n_hidden, n_classes))
        else:
            self.layers.append(SAGEConv(in_feats, n_classes))
        self.activation = activation
        self.dropout = dropout
    
    def forward(self, blocks, feat):
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
            
def run(args, device):
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    train_g = val_g = test_g = g
    train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
    train_labels = val_labels = test_labels = g.ndata.pop('labels')
    n_classes = dataset.num_classes
    in_feat = train_nfeat.shape[1]
    
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    
    dataloader_device = torch.device('cpu')
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device
    
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fanout.split(",")])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device = dataloader_device,
        batch_size= args.batchsize,
        shuffle = True
    )
    model = Model(in_feat, args.num_hidden, n_classes, args.num_layers,F.relu, F.dropout)
    model = model.to(device)
    loss_fuc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    
    for epoch in range(args.num_epochs):
        for iter,
        
    
    
    
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    # argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    args = argparse.parse_args()
    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')
    
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]
    n_classes = dataset.num_classes
    