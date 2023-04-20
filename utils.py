import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()    
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print ('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.num_classes = args.way
    if args.dataset == 'MiniImageNet':
        args.num_class = 64  # mini
        args.coarse_class = 8  # mini
    elif 'Tiered' in args.dataset:
        args.num_class = 351
        args.coarse_class = 20
    train_way = args.way
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(train_way, args.shot, args.query)])
    save_path2 = '_'.join(['total_epoch{}'.format(args.max_epoch), str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler), 
                           'T1{}T2{}'.format(args.temperature, args.temperature2),
                           'b{}'.format(args.balance),
                           'bsz{:03d}'.format( max(args.way, args.num_classes)*(args.shot+args.query) ),

                           ])    
    if args.init_weights is not None:
        save_path1 += '-Pre'
    if args.use_euclidean:
        save_path1 += '-DIS'
    else:
        save_path1 += '-SIM'
    save_path1 += str(args.pre_name)


    if args.fix_BN:
        save_path2 += '-FBN'
    if not args.augment:
        save_path2 += '-NoAug'
    if args.cos:
        save_path2 += '-cosine'
    save_path2 += '-'
    save_path2 += str(args.name)
    if args.fix_pre:
        save_path2 += '-fix_pre'
            
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--test_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='FEAT', 
                        choices=['protonet', 'mgc'])
    parser.add_argument('--use_euclidean', action='store_true', default=False)    
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet'])
    parser.add_argument('--open_loss_type', type=str, default='MaxProbability',
                        choices=['MaxProbability', 'BinaryClassification'])
    
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--eval_open_way', type=int, default=5)
    parser.add_argument('--balance', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=64)
    parser.add_argument('--neg_type', type=str, default='mean', choices=['mean', 'min', 'max', 'mlp', 'attention'])
    parser.add_argument('--coarse_type', type=str, default='attention', choices=['cross', 'attention', 'att', 'att_and_att'])
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--aux', action='store_true', default=False) 
    parser.add_argument('--consistant', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=0.5) 
    parser.add_argument('--alpha', type=float, default=0.5) 
    parser.add_argument('--drop_rate', type=float, default=0.5) 
    parser.add_argument('--drop_rate_buquan', type=float, default=0.5) 
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--logits', action='store_true', default=False) 
    parser.add_argument('--test_logits', action='store_true', default=False) 
    parser.add_argument('--ab_all', action='store_true', default=False) 
    parser.add_argument('--enproto_trans', action='store_true', default=False) 
    parser.add_argument('--with_itself', action='store_true', default=False) 
    parser.add_argument('--bcls_score', action='store_true', default=False)
    parser.add_argument('--bcls_embeddding', action='store_true', default=False)
    parser.add_argument('--only_bcls_logits', action='store_true', default=False)
    parser.add_argument('--pre_train_cosine', action='store_true', default=False)
    parser.add_argument('--biased', action='store_true', default=False)
    parser.add_argument('--add_only', action='store_true', default=False)
    parser.add_argument('--need_coarse', action='store_true', default=False) 
    parser.add_argument('--source_logits', action='store_true', default=False)
    parser.add_argument('--buquan_coarse_instance', action='store_true', default=False)
    parser.add_argument('--test_balance', action='store_true', default=False)
    parser.add_argument('--naive', action='store_true', default=False)
    parser.add_argument('--first_proto', action='store_true', default=False)
    parser.add_argument('--sup_itself', action='store_true', default=False)
    parser.add_argument('--selfsup', action='store_true', default=False)
    parser.add_argument('--only_coarse', action='store_true', default=False)
    parser.add_argument('--gating', action='store_true', default=False)
    parser.add_argument('--base_weight', action='store_true', default=False)
    parser.add_argument('--pretrained_model_path', type=str, default=None)

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine', 'plateau'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.2)    
    parser.add_argument('--fix_BN', action='store_true', default=False)     # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--cos', action='store_true', default=False)
    parser.add_argument('--fix_pre', action='store_true', default=False)
    parser.add_argument('--fix_knowledge', action='store_true', default=False) 
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--slf_flag_norm', action='store_true', default=False) 
    parser.add_argument('--slf_att_norm', action='store_true', default=False) 
    parser.add_argument('--buquan_norm', action='store_true', default=False)
    parser.add_argument('--buquan_coarse', action='store_true', default=False)
    parser.add_argument('--dev', action='store_true', default=False)
    parser.add_argument('--dev_value', type=float, default=1.)
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='max_acc_sim')
    parser.add_argument('--pre_name', type=str, default='')

    parser.add_argument('--levels', type=int, default=2)
    
    return parser
