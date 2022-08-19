import torch
import torch.nn as nn
import numpy as np


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5) 
        else:
            raise ValueError('')

        if args.pre_train_cosine:
            weight_base = torch.FloatTensor(args.num_class, hdim).normal_(
                0.0, np.sqrt(2.0 / hdim))
            self.weight_base = nn.Parameter(weight_base, requires_grad=True)
            scale_cls =  10.0
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls),
                requires_grad=True)

            weight_coarse = torch.FloatTensor(args.coarse_class, hdim).normal_(
                0.0, np.sqrt(2.0 / hdim))
            self.weight_coarse = nn.Parameter(weight_coarse, requires_grad=True)
            scale_cls_coarse =  10.0
            self.scale_cls_coarse = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls_coarse),
                requires_grad=True)
            if args.biased:
                self.bias_coarse = nn.Parameter(
                    torch.FloatTensor(1).fill_(0), requires_grad=True)
                self.bias = nn.Parameter(
                    torch.FloatTensor(1).fill_(0), requires_grad=True)
        else:
            self.fc = nn.Linear(hdim, args.num_class)
            if args.levels == 2:
                args.coarse_class = 8
                if 'iere' in args.dataset:
                    args.coarse_class = 20

            self.fc_coarse = nn.Linear(hdim, args.coarse_class)



    def split_instances(self, num_inst):
        args = self.args
        if True:
            if self.training:
                return (torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
                        torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.query))).long().view(
                            1, args.query, args.way),
                        torch.Tensor(np.arange(args.way * (args.shot + args.query), num_inst)).long().view(
                            1, args.query, args.way)
                        )
            else:
                return (
                torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way),
                torch.Tensor(np.arange(args.eval_way * args.eval_shot,
                                       args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way),
                torch.Tensor(np.arange(args.eval_way * (args.eval_shot + args.eval_query), num_inst)).long().view(1, args.eval_query, args.eval_open_way)
                )
    def forward(self, x, label, get_feature=False, testing=False, cls_params=None):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0] 
            support_idx, query_idx, open_idx = self.split_instances(num_inst)
            if self.args.model_class == 'mgc' or 'level' in self.args.model_class:
                logits, logits_reg, cls_logits, bcls_logits, plain_proto, enhance_proto, instance_logits = self._forward(instance_embs, support_idx, query_idx, open_idx, label, testing, cls_params)
                return logits, logits_reg, cls_logits, bcls_logits, plain_proto, enhance_proto, instance_logits
            else:
                logits, logits_reg, cls_logits, bcls_logits = self._forward(instance_embs, support_idx, query_idx, open_idx, label, testing, cls_params)
                return logits, logits_reg, cls_logits, bcls_logits

    def _forward(self, x, support_idx, query_idx, open_idx, label, testing, cls_params):
        raise NotImplementedError('Suppose to be implemented by subclass')