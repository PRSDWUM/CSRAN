# -----------------------------------------------------------
# Generative Label Fused Network implementation based on
# Position Focused Attention Network (PFAN) and Stacked Cross Attention Network (SCAN)
# the code of PFAN: https://github.com/HaoYang0123/Position-Focused-Attention-Network
# the code of SCAN: https://github.com/kuanghuei/SCAN
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from GAT_models import GAT
import torch.nn.functional as F


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx,dim=1)

    r_inv_sqrt = rowsum.pow(-0.5).flatten(0)
   
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    
    result = torch.matmul(mx,r_mat_inv_sqrt).t()
    return torch.matmul(result,r_mat_inv_sqrt)
    


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx,dim=1)
    
    r_inv = rowsum.pow(-1).flatten(0)
    
    r_inv[torch.isinf(r_inv)] = 0.
    
    r_mat_inv = torch.diag(r_inv)
    
    mx = torch.matmul(r_mat_inv,mx)
   
    return mx



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(img_dim, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer，
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)


    def forward(self, images):
        features = self.fc(images)
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)
        return features


    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)



class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, embed_size, num_layers, bert_to_gru_size, use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.use_bi_gru = use_bi_gru
        self.fc = nn.Linear(768, bert_to_gru_size)
        self.rnn = nn.GRU(bert_to_gru_size, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        # self.embed.weight.data.uniform_(-0.1, 0.1)
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)


    def forward(self, x, lengths):
        """Handles variable size captions
        """
        x = self.fc(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True) # 
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
       
        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] + cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        if not self.no_txtnorm:
            cap_emb_final = l2norm(cap_emb, dim=-1)

        return cap_emb_final, cap_len


def func_attention(query, context, opt, smooth):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)


        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """

    def __init__(self, opt):

        self.grad_clip = opt.grad_clip

        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.embed_size, opt.num_layers, opt.bert_to_gru_size, use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        self.use_GAT = opt.use_GAT

        if opt.use_GAT:
            self.GAT_model = GAT(nfeat=1024,
                 nhid=opt.GAT_hidden,
                 nclass=opt.GAT_nclass,
                 dropout=opt.GAT_dropout,
                 nheads=opt.GAT_nb_heads,
                 alpha=opt.GAT_alpha)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            if opt.use_GAT:
                self.GAT_model.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        if opt.use_GAT:
            params += list(self.GAT_model.parameters())

        self.params = params
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate,eps=1e-08, weight_decay=0)
#self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate) 原来文章里的都是这个设置？
        self.Eiters = 0

    def state_dict(self, opt):
        if opt.use_GAT:
            state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.GAT_model.state_dict()]
        else:
            state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        # state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        if self.use_GAT:
            self.GAT_model.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        if self.use_GAT:
            self.GAT_model.train()


    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        if self.use_GAT:
            self.GAT_model.eval()

    def forward_emb(self, opt, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()


        img_emb0 = self.img_enc(images)
        cap_emb0, cap_lens = self.txt_enc(captions, lengths)

        if opt.use_GAT:
            batch_size = img_emb0.shape[0]
            img_emb = img_emb0.view(batch_size*36, 1024)
            cap_emb = cap_emb0.reshape(batch_size*cap_lens[0], 1024)

            self.features = torch.cat((img_emb,cap_emb),dim=0)
            self.adj = torch.zeros([batch_size*(36+cap_lens[0]),batch_size*(36+cap_lens[0])])
            if opt.train_dev_log.startswith("train"):

            #     # 
                # for i in range(batch_size):
                #     self.adj[i*36:(i+1)*36,i*36:(i+1)*36] = 1
                # 
                start_index = batch_size*36
                # for j in range(batch_size):
                #     self.adj[start_index+j*cap_lens[0]:start_index+j*cap_lens[0]+cap_lens[j],start_index+j*cap_lens[0]:start_index+j*cap_lens[0]+cap_lens[j]] = 1
#                 # #

                for k in range(batch_size):
                    self.adj[k*36:(k+1)*36,start_index+k*cap_lens[0]:start_index+k*cap_lens[0]+cap_lens[k]] = 1
                    self.adj[start_index+k*cap_lens[0]:start_index+k*cap_lens[0]+cap_lens[k],k*36:(k+1)*36] = 1
                # one = torch.ones_like(self.adj)
                # zero = one - 1
                # if torch.cuda.is_available():
                #     one = one.cuda()
                #     zero = zero.cuda()
                # cossim = torch.matmul(self.features,self.features.T)
                # self.adj = torch.where(cossim > 0, one, zero)

            else:

                one = torch.ones_like(self.adj)
                zero = one - 1
                if torch.cuda.is_available():
                    one = one.cuda()
                    zero = zero.cuda()
                cossim = torch.matmul(self.features,self.features.T)
                self.adj = torch.where(cossim > 0, one, zero)
                # print('dev or test self.adj value is:')
                # print(self.adj)





            self.features, self.adj = Variable(self.features), Variable(self.adj)
            if torch.cuda.is_available():
                self.features = self.features.cuda()
                self.adj = self.adj.cuda()

    #         self.features = normalize_features(self.features)
    #         self.adj = normalize_adj(self.adj)
            opt.GAt_ouput = self.GAT_model(self.features, self.adj, opt.train_dev_log)



            img_emb = opt.GAt_ouput[0:batch_size*36]
            cap_emb = opt.GAt_ouput[batch_size*36:]
            img_emb_GAT = img_emb.view(batch_size, 36, 1024)
            img_emb_GAT = l2norm(img_emb_GAT, dim=-1)
            cap_emb_GAT = cap_emb.view(batch_size, cap_lens[0], 1024)
            cap_emb_GAT = l2norm(cap_emb_GAT, dim=-1)
            # # only with GAT
            # img_emb =img_emb_GAT
            # cap_emb =cap_emb_GAT

            img_emb =torch.cat((img_emb_GAT,img_emb0),dim=2)
            cap_emb =torch.cat((cap_emb_GAT,cap_emb0),dim=2)
        else:
            img_emb =img_emb0
            cap_emb =cap_emb0

        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, opt, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb, cap_lens = self.forward_emb(opt, images,captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)
        loss.backward()        
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)      
        self.optimizer.step()


       
