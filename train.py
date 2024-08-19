import tensorboard_logger as tb_logger
import os
import time
import torch
import numpy
import data as data
from model import SCAN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i, shard_xattn_i2t, evalrank, evaluation_ensemble
import logging
import argparse
from CMRsystem import display_main
from test import test_main



def save_path(opt):
    save_parameter_path = 'MSCOCO_adj_0and1'
    dataset_name = opt.data_name
    t2iori2t = opt.cross_attn
    embed_size = opt.embed_size
    num_epochs = opt.num_epochs
    folder_name = dataset_name + '_' + t2iori2t + '_' + str(embed_size) + '_' + str(num_epochs)
    path = os.path.join(save_parameter_path, folder_name)
    return path, folder_name

def main():
    # Hyper Parameters
    print(torch.cuda.is_available())
    print("Start for main")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--lr_update', default=8, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=300, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=18, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--max_violation', default=True,
                         help='Use max instead of sum in the rank loss.') 
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--raw_feature_norm', default="l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--precomp_enc_type', default="basic",
                        help='basic|weight_norm')
    parser.add_argument('--bi_gru', default=True,
                         help='Use bidirectional GRU.')
    parser.add_argument('--use_GAT', default=True,
                         help='Use GAT.')

    parser.add_argument('--val_step', default=50000000000, type=int,
                        help='Number of steps to run validation.')


    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')   
    parser.add_argument('--agg_func', default="Mean",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--lambda_lse', default=6., type=float,
                        help='LogSumExp temp.')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--bert_to_gru_size', default=300, type=int, help='bert_to_gru_size')
    parser.add_argument('--drop_out', default=0.5, type=float, help='drop_out')
    # 
    parser.add_argument('--GAT_lr', type=float, default=0.0002, help='Initial learning rate.')
    parser.add_argument('--GAT_weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--GAT_nclass', type=int, default=1024, help='Number of output class.')
    parser.add_argument('--GAT_dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--GAT_alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--GAT_hidden', type=int, default=1024, help='Number of hidden units.')
    parser.add_argument('--GAT_nb_heads', type=int, default=4, help='Number of head attentions.')


    opt = parser.parse_args()
    current_path, folder_name = save_path(opt)
    if not os.path.exists(current_path):
        os.mkdir(current_path)
    opt.current_path = current_path
    opt.model_name = opt.current_path
    opt.logger_name = opt.current_path
    print(opt)

    global mode_name
    mode_name = opt.cross_attn
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    print('loading')
    train_loader, val_loader = data.get_loaders(
        opt.data_name, opt.batch_size, opt.workers, opt)
    print('load done')

    model = SCAN(opt)


    best_rsum = 0
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)
        opt.train_dev_log = None
        train(opt, train_loader, model, epoch, val_loader)

        model.val_start()
        rsum = validate(opt, val_loader, model)
        is_best = rsum > best_rsum
        print('is the best result -{0}，the current result is: {1}，the best result is: {2}.'.format(is_best, rsum, best_rsum))
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(opt),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, opt,
            filename='checkpoint_{}_posiAttn_' + '.pth.tar'.format(mode_name,epoch),
            prefix=opt.model_name + '/')

    input_data_name = input('dev_results: choose the dataset name(f30k_precomp or coco_precomp):')
    if (input_data_name == 'f30k_precomp'):
        data_path = 'data/f30k_precomp'
        dataset_name = 'f30k-images'
        split_name = 'dev'
        times = 5
        display_main(opt.model_name, data_path, dataset_name, split_name, times)
    else:
        data_path = 'data/coco_precomp'
        dataset_name = 'coco-images'
        split_name = 'dev'
        times = 5
        display_main(opt.model_name, data_path, dataset_name, split_name, times)
        
    
        

    print('--------------------------------------------:')
    print('--------------finally_test------------------:')
    test_flag = input('go into the test system(y or n):')
    if (test_flag == 'y'):
        ensemble_flag = input('evaluation ensemble or single(y or n):')
        fold5_flag = input('whether use the fold5 cross validation or not(y or n)')
        if (fold5_flag == 'y'):
            fold5 = True
        else:
            fold5 = False
        if(ensemble_flag == 'y'):
            prefix = os.path.join(os.path,'best_models/')
            model1_path = prefix + 'model_best_i2t_GAT_'.format(mode_name) + '.pth.tar'
            model2_path = prefix + 'model_best_t2i_GAT_'.format(mode_name) + '.pth.tar'
            data_path = os.path.join(opt.data_path, opt.data_name)
            split = 'test'
            fold5 = fold5
            evaluation_ensemble.evalrank(model1_path, model2_path, data_path, split, fold5)
        else:
            #
            prefix = opt.model_name + '/'
            model_path = prefix + 'model_best_{}_GAT_'.format(mode_name) + '.pth.tar'
            data_path = os.path.join(opt.data_path, opt.data_name)
            split = 'test'
            fold5 = fold5
            evalrank(model_path, data_path, split, fold5)
        
    else:
        print('The whole work has done!')
        
    input_data_name = input('test_results: choose the dataset name(f30k_precomp or coco_precomp):')
    if (input_data_name == 'f30k_precomp'):
        data_path = 'data/f30k_precomp'
        dataset_name = 'f30k-images'
        split_name = 'test'
        times = 5
        display_main(opt.model_name, data_path, dataset_name, split_name, times)
    else:
        data_path = 'data/coco_precomp'
        dataset_name = 'coco-images'
        split_name = 'test'
        times = 5
        display_main(opt.model_name, data_path, dataset_name, split_name, times)
        
    
def train(opt, train_loader, model, epoch, val_loader):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    opt.train_dev_log = "train"

    end = time.time()
    for i, train_data in enumerate(train_loader):

        # switch to train mode
        '''image, whole, box, caption, length, temp = train_data
        if max_len <caption.size(1):
            max_len = caption.size(1)
            print(max_len)
        '''
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)
        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(opt, *train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} \t'
                'Data {data_time.val:.3f} \t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters) # 写入日志文件

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)
            

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    opt.train_dev_log = "dev"
    img_embs, cap_embs, cap_lens = encode_data(
        opt, model, val_loader, opt.log_step, logging.info)
    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    print("Img shape in validate:", img_embs.shape)

    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(opt.model_name, opt.train_dev_log, img_embs, cap_embs, cap_lens, sims, opt)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(opt.model_name, opt.train_dev_log, 
        img_embs, cap_embs, cap_lens, sims, opt)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, opt, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if is_best:
                file_name = prefix + 'model_best_{}_GAT_'.format(mode_name) + '.pth.tar'
                torch.save(state, file_name)
                print('Best Model parameters Update!')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
