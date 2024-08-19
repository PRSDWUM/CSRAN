import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_layers import GraphAttentionLayer

def l2norm_GAT(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """ # 下面是对X进行逐个元素的求平方，然后第二维度进行求和，保持维度不变，然后开根号+小数值
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm) # 进行归一化操作
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
        # x = F.dropout(x, self.dropout, training=self.training) #在x的计算过程中，需要使用dropout，因为x是变动的，所以这样会更好？然后显示告诉是traiinging阶段，可能会自动进行缩放处理？training属性受train()和eval()方法影响而改变
        x = torch.cat([att(x, adj, split_name) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj,split_name))
        x = F.log_softmax(x, dim=1)
        return x


