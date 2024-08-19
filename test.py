import os
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_xattn_t2i, shard_xattn_i2t, evalrank, evaluation_ensemble 
from CMRsystem import display_main
import argparse


def test_main(opt):
    print('--------------------------------------:')
    print('--------------Test begin--------------:')
    
    test_flag = input('go into the test system(y or n):')
    if (test_flag == 'y'):
        ensemble_flag = input('evaluation ensemble or single(y or n):')
        fold5_flag = input('whether use the fold5 cross validation or not(y or n)')
        if (fold5_flag == 'y'):
            fold5 = True
        else:
            fold5 = False
        
        if(ensemble_flag == 'y'):
            path = os.getcwd()
            print(path)
            prefix = os.path.join(path,'MSCOCO_adj_0and1_0128')
            model1_path = prefix + '/model_best_i2t_GAT_.pth.tar'
            model2_path = prefix + '/model_best_t2i_GAT_.pth.tar'
            data_path = os.path.join(opt.data_path, opt.data_name)
            split = 'testall'
            fold5 = fold5
            evaluation_ensemble(model1_path, model2_path, data_path, split, fold5)
        else:
            #单向模型验证
            model_name = opt.cross_attn
            prefix = opt.model_name + '/'
            model_path = prefix + 'model_best_{}_GAT_'.format(model_name) + '.pth.tar'
            data_path = os.path.join(opt.data_path, opt.data_name)
            split = 'testall'
            fold5 = fold5
            evalrank(model_path, data_path, split, fold5)
        
    else:
        print('The whole work has done!')
        
   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='coco_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--num_epochs', default=18, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    opt = parser.parse_args()
    save_parameter_path = 'MSCOCO_adj_0and1_0128'
    folder_name = opt.data_name + '_' + opt.cross_attn + '_' + str(opt.embed_size) + '_' + str(opt.num_epochs)
    opt.model_name = os.path.join(save_parameter_path, folder_name)
    test_main(opt)
    
    
    
 
    
