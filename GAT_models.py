import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_layers import GraphAttentionLayer

def l2norm_GAT(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """ # 
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm) # 
    return X

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        
    def forward(self, x, adj, split_name):
        # x = F.dropout(x, self.dropout, training=self.training) #
        x = torch.cat([att(x, adj, split_name) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj,split_name))
        x = F.log_softmax(x, dim=1)
        return x


