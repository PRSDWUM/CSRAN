# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# 注意，对于 MSCOCO 而言，若测试集包含 5,000 张图像，一般设置两种测试方式：

# MSCOCO1K，即将5000张图像划分为5部分，分别包含1000张图像，最终测试结果为在5个测试集结果的平均值。
# MSCOCO5K，直接对5000张图像进行测试
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os
import pickle
import sys
from data import get_test_loader
import time
import numpy as np
import torch
from model import SCAN, xattn_score_t2i, xattn_score_i2t
from collections import OrderedDict
import time
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(opt, model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(opt, images, captions, lengths, volatile=True)
        #print(img_emb)
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids, :, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, cap_len)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs, cap_lens


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.train_dev_log = "test"
    print(opt)
    if data_path is not None:
        opt.data_path = data_path


    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(opt, model, data_loader)
    
    
    
    
    
#     #下面为tsne需要代码--1008
#     #-------------------------------------------------------------
#     print('tsne原始数据形状为：')
#     print(img_embs.shape)
#     print(cap_embs.shape)
#     image_tsne = image_embs.view(opt.batch_size, 36, -1).sum(dim=2)
#     cap_tsne = cap_embs.view(opt.batch_size, cap_len[0], -1).sum(dim=2)
#     print('tsne测试')
#     print(image_tsne.shape)
#     print(cap_tsne.shape)
#     with open('coco-features.txt','a') as f:
#         for item in image_tsne:
#             print(item, end='\n', file=f)
#         for item in cap_tsne:
#             print(item, end='\n', file=f)
#     print('coco的特征以被保存，以一张图像为单位')
#     #-------------------------------------------------------------
    
    
    
    
    
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))


    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        else:
            raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end-start)

        r, rt = i2t(os.path.dirname(model_path), split, img_embs, cap_embs, cap_lens, sims, fold5, return_ranks=True)
        ri, rti = t2i(os.path.dirname(model_path), split, img_embs, cap_embs, cap_lens, sims, fold5, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO------only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            start = time.time()
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            else:
                raise NotImplementedError
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(os.path.dirname(model_path), split, img_embs_shard,cap_embs_shard, cap_lens_shard, sims, fold5, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(os.path.dirname(model_path), split, img_embs_shard, cap_embs_shard, cap_lens_shard, sims, fold5, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12] * 5))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def evaluation_ensemble(model_path, model_path2, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    checkpoint2 = torch.load(model_path2)
    opt2 = checkpoint2['opt']

    print(opt)
    print(opt2)
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = SCAN(opt)
    model2 = SCAN(opt2)

    # load model state
    model.load_state_dict(checkpoint['model'])

    model2.load_state_dict(checkpoint2['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name,
                                  opt.batch_size, opt.workers, opt)

    start_total = time.time()
    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(opt, model, data_loader)
    img_embs2, cap_embs2, cap_lens2 = encode_data(opt2, model2, data_loader)

    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:
        # no cross-validation, full evaluation
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        img_embs2 = np.array([img_embs2[i] for i in range(0, len(img_embs2), 5)])

        start = time.time()
        
        # # 原版方案一
        # if opt.cross_attn == 't2i':
        #     sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        #     sims2 = shard_xattn_t2i(img_embs2, cap_embs2, cap_lens2, opt2, shard_size=128)
        # elif opt.cross_attn == 'i2t':
        #     sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        #     sims2 = shard_xattn_i2t(img_embs2, cap_embs2, cap_lens2, opt2, shard_size=128)
        # else:
        #     raise NotImplementedError
            
        # 改进方案二
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        sims2 = shard_xattn_t2i(img_embs2, cap_embs2, cap_lens2, opt2, shard_size=128)
        
        
        
        end = time.time()
        print("calculate similarity time:", end - start)

        sims = (sims + sims2) / 2

        r, rt = i2t(os.path.dirname(model_path), split, img_embs, cap_embs, cap_lens, sims, fold5, return_ranks=True)
        ri, rti = t2i(os.path.dirname(model_path), split, img_embs, cap_embs, cap_lens, sims, fold5, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            img_embs_shard2 = img_embs2[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard2 = cap_embs2[i * 5000:(i + 1) * 5000]
            cap_lens_shard2 = cap_lens2[i * 5000:(i + 1) * 5000]

            start = time.time()
# 原版方案一：           
#             if opt.cross_attn == 't2i':
#                 sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
#                 sims2 = shard_xattn_t2i(img_embs_shard2, cap_embs_shard2, cap_lens_shard2, opt2, shard_size=128)
#             elif opt.cross_attn == 'i2t':
#                 sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
#                 sims2 = shard_xattn_i2t(img_embs_shard2, cap_embs_shard2, cap_lens_shard2, opt2, shard_size=128)
#             else:
#                 raise NotImplementedError
            
            # 改进方案二
            sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            sims2 = shard_xattn_t2i(img_embs_shard2, cap_embs_shard2, cap_lens_shard2, opt2, shard_size=128)
            
            
            end = time.time()
            print("calculate similarity time:", end - start)

            sims = (sims + sims2) / 2

            r, rt0 = i2t(os.path.dirname(model_path), split, img_embs_shard, cap_embs_shard, cap_lens_shard, sims, fold5, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(os.path.dirname(model_path), split, img_embs_shard, cap_embs_shard, cap_lens_shard, sims, fold5, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[12] * 5))
        print("Average i2t Recall: %.1f" % mean_metrics[10])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[11])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')

    end_total = time.time()
    print('test time (S): ' + str(end_total - start_total))



def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(int(n_cap_shard)):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            with torch.no_grad():
                if torch.cuda.is_available():
                    im = Variable(torch.from_numpy(images[im_start:im_end])).cuda()
                    s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                else:
                    im = Variable(torch.from_numpy(images[im_start:im_end]))
                    s = Variable(torch.from_numpy(captions[cap_start:cap_end]))
            l = caplens[cap_start:cap_end]
            sim = xattn_score_t2i(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(int(n_im_shard)):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(int(n_cap_shard)):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            with torch.no_grad():
                if torch.cuda.is_available():
                    im = Variable(torch.from_numpy(images[im_start:im_end])).cuda()
                    s = Variable(torch.from_numpy(captions[cap_start:cap_end])).cuda()
                else:
                    im = Variable(torch.from_numpy(images[im_start:im_end]))
                    s = Variable(torch.from_numpy(captions[cap_start:cap_end]))
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t(path, split, images, captions, caplens, sims, fold5, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    log_1_or_5k = ''
    if fold5:
        log_1_or_5k = '_1k'
    else:
        log_1_or_5k = '_5k'
        
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    new_txt_file = '%s_i2t%s.txt' % (split, log_1_or_5k)
    new_txt_file_path = os.path.join(path, new_txt_file)
    f = open(new_txt_file_path, "a")
    f.truncate(0)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        count = 0
        for inds_item in inds:
            if (count<15):
                print(inds_item, end=" ", file=f)
            else:
                break;
            count  = count + 1
        print('', file=f)
        # f.write(str(inds)+'\n')
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
    f.close()
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(path, split, images, captions, caplens, sims, fold5, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    log_1_or_5k = ''
    if fold5:
        log_1_or_5k = '_1k'
    else:
        log_1_or_5k = '_5k'
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    new_txt_file = '%s_t2i%s.txt' % (split, log_1_or_5k)
    new_txt_file_path = os.path.join(path, new_txt_file)
    f = open(new_txt_file_path, "a")
    f.truncate(0)
    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]
            count = 0
            for inds_item in inds:
                if (count<15):
                    print(inds_item, end=" ", file=f)
                else:
                    break;
                count  = count + 1
            print('', file=f)
    f.close()
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)