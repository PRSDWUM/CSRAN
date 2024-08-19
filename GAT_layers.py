import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
# 
#     def forward(self, h, adj, split_name):
#         Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
#         e = self._prepare_attentional_mechanism_input(Wh)
#         
#         zero_vec = -9e15*torch.ones_like(e)
#         zero = torch.ones_like(e)-1
#         if split_name.startswith("train"):
#             thre = 0
#             attention = torch.where(adj > thre, e, zero_vec)
#             

#         else:
#             thre = 0.4
#             attention = torch.where(e > thre, e, zero_vec)
#             adj = torch.where(e > thre, torch.ones_like(e), zero)
            

#         attention = F.softmax(attention, dim=1)
#         attention1 = F.dropout(attention, self.dropout, training=self.training)
#         attention = torch.where(adj>thre,attention1,zero)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
    def forward(self, h, adj, split_name):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        zero = torch.ones_like(e)-1
#         if split_name.startswith("train"):
#             attention = torch.where(adj > 0, e, zero_vec)
#             

#         else:
#             attention = e
#             #adj = torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
    
        attention1 = F.softmax(attention, dim=1)
        attention2 = F.dropout(attention1, self.dropout, training=self.training)
        attention3 = torch.where(adj > 0, attention2, zero)# 只有这一句是变动的了
        h_prime = torch.matmul(attention3, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
