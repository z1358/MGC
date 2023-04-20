import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler, RandomSampler, ClassSampler

class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])
                
                yield ( torch.cat(_, dim=0) for _ in output_batch )
            except StopIteration:
                done = True
        return

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'TieredImageNet':
        from dataloader.tiered_imagenet import tieredImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    trainset = Dataset('train', args, augment=args.augment)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,
                                      max(args.way, args.num_classes),
                                      args.shot + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)


    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            args.eval_way, args.eval_shot + args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                            args.test_episodes,
                            args.eval_way, args.eval_shot + args.eval_query, n_open=args.eval_open_way)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)    

    return train_loader, val_loader, test_loader

def prepare_model(args):
    model = eval(args.model_class)(args)
    for k, v in model.named_parameters():
        print(k)


    if (args.init_weights is not None) and (args.init_weights != 'test'):
        print("pre-train init loading!")
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights)['params']
        for k, v in pretrained_dict.items():
            print(k)


        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        if 'ered' in args.dataset:
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

        other_dict = {k: v for k, v in pretrained_dict.items() if k not in model_dict}
        print('other_dict {}'.format(other_dict.keys()))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)
    if args.fix_pre:
        for para in model.encoder.parameters():
            para.requires_grad = False
        for para in model.fc.parameters():
            para.requires_grad = False
        for para in model.fc_coarse.parameters():
            para.requires_grad = False

    if args.fix_knowledge:
        for para in model.fc.parameters():
            para.requires_grad = False
        for para in model.fc_coarse.parameters():
            para.requires_grad = False

    return model, para_model

def prepare_optimizer(model, args):

    top_para_name = [k for k, v in model.named_parameters() if (('slf' in k) or ('neg' in k) or ('binarycls' in k) or ('proto' in k))]
    slow_para_name = [k for k, v in model.named_parameters() if k not in top_para_name]

    requires_grad_names = []
    for name, param in model.named_parameters():
        if param.requires_grad and (name in slow_para_name):
            requires_grad_names.append(name)

    requires_grad_names_fast = []
    for name, param in model.named_parameters():
        if param.requires_grad and (name in top_para_name):
            requires_grad_names_fast.append(name)


    top_para = [v for k, v in model.named_parameters() if k in top_para_name]
    slow_para = [v for k, v in model.named_parameters() if k in slow_para_name]

    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, model.encoder.parameters())},
             {'params': filter(lambda p: p.requires_grad, model.fc.parameters())},
             {'params': filter(lambda p: p.requires_grad, model.fc_coarse.parameters())},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    else:
        optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, slow_para)},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
