import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from models import FewShotModel
import os.path as osp
import random

class MlpNeg(nn.Module):
    def __init__(self, k, feat_dim, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0.0:
            self.head = nn.Sequential(
                nn.Linear(k * feat_dim, 1024),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, feat_dim)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(k * feat_dim, 1024),
                nn.LeakyReLU(0.1),
                nn.Linear(1024, feat_dim)
            )

    def forward(self, input):
        way = input.shape[0]
        input = input.view(way, -1)
        out = self.head(input)
        return out


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class Cross(nn.Module):
    ''' cross network '''

    def __init__(self, feat_dim, layer=1):
        super().__init__()
        self.corss_first = nn.Linear(feat_dim, 1)

    def forward(self, coarse_proto, task_proto):
        residual = task_proto
        out = self.corss_first(torch.bmm(task_proto.permute([0, 2, 1]), coarse_proto))
        out = out.permute([0, 2, 1])
        out = out + residual
        return out
class CrossNet(nn.Module):

    def __init__(self, in_features, layer_num=2, parameterization='vector'):
        super().__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])


    def forward(self, coarse_proto, task_proto):
        coarse_proto = coarse_proto.squeeze(1)
        task_proto = task_proto.squeeze(1)
        x_0 = coarse_proto.unsqueeze(2)
        x_l = task_proto.unsqueeze(2)
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        x_l = torch.unsqueeze(x_l, dim=1) # 5 1 640
        return x_l

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, args, n_head, d_model, d_k, d_v, dropout=0.1, trans=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.flag_norm = args.slf_flag_norm
        self.slf_att_norm = args.slf_att_norm
        self.buquan_norm = args.buquan_norm
        self.trans = trans
        self.add = args.add_only

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_q, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_q, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        if not self.trans:
            if self.flag_norm:
                output = self.layer_norm(output)
            else:
                output = output
        elif self.trans:
            if self.add:
                output = output + residual
            elif self.buquan_norm:
                output = self.layer_norm(output + residual)
            else:
                output = output
        output = output.squeeze(1)
        return output


class MGC(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        elif args.backbone_class == 'WRN':
            hdim = 640
        else:
            raise ValueError('')

        self.coarse = self.coarse_cluster()
        self.k = args.topk

        if args.buquan_norm:
            self.slf_att = MultiHeadAttention(args, 1, hdim, hdim, hdim, dropout=args.drop_rate_buquan, trans=True)
        if args.enproto_trans:
            self.enproto_trans = MultiHeadAttention(args, 1, hdim, hdim, hdim, dropout=args.drop_rate, trans=True)
        if args.buquan_coarse:
            if args.coarse_type == 'cross':
                self.slf_coarse = CrossNet(hdim, layer_num=args.layers)
                # self.slf_coarse = Cross(hdim)
            else:
                self.slf_coarse = MultiHeadAttention(args, 1, hdim, hdim, hdim, dropout=args.drop_rate_buquan,
                                                     trans=True)

    def coarse_cluster(self):
        netdict = {}
        THIS_PATH = osp.dirname(__file__)
        ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
        if self.args.dataset == 'MiniImageNet':
            max_label = 8
            SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
        elif (self.args.dataset == 'TieredImageNet' or self.args.dataset == 'TieredImagenet'):
            max_label = 20
            SPLIT_PATH = osp.join(ROOT_PATH, 'data/tieredimagenet/split')
        with open(osp.join(SPLIT_PATH, 'coarse.txt')) as words:
            ll = -1
            for line in words:
                ll += 1
                netdict[ll] = line.strip('\n').split('\t')[1]
        cluster = []
        for i in range(max_label):
            coarse = []
            for j in netdict:
                if int(netdict[j]) == int(i):
                    coarse.append(j)
            cluster.append(coarse)
        return cluster

    def find_topk(self, score, k, label=None, testing=False):
        coarse_matrix = torch.zeros([self.args.way, len(self.coarse)]).cuda()
        indexs_topk_matrix = torch.zeros([len(self.coarse), self.args.way, k]).long().cuda()

        if not testing:
            index = (
                torch.LongTensor([number for number in range(self.args.way)]).cuda(),
                torch.LongTensor([label[number] for number in range(self.args.way)]).cuda(),
            )
            new_value = torch.Tensor([-1.0]).cuda()
            score.index_put_(index, new_value)

        for i in range(len(self.coarse)):
            idx_tensor = torch.tensor(self.coarse[i])
            idx_score = score[:, idx_tensor].contiguous().view(self.args.way,
                                                               -1)  # support_idx.contiguous().view(-1)].contiguous()
            idx_score_sorted, indexs = torch.sort(idx_score, descending=True)
            coarse_topk = idx_score_sorted[:, :k]
            coarse_matrix[:, i] = torch.sum(coarse_topk, dim=1)
            indexs_topk = indexs[:, :k]
            index_to_label = idx_tensor[indexs_topk]
            indexs_topk_matrix[i, :, :] = index_to_label
        coarse_label = coarse_matrix.argmax(dim=1)
        return coarse_label, indexs_topk_matrix

    def extra_weight(self, belong_cluster, topk_idx):
        base_data = []
        for i in range(len(belong_cluster)):
            idx = topk_idx[i, :]
            idx_weight = self.fc.weight[idx, :].contiguous().view(1, self.k, -1)
            if self.args.need_coarse:
                coarse_proto = self.fc_coarse.weight[belong_cluster[i], :].contiguous().view(1, -1, 640)
                idx_weight = torch.cat([idx_weight, coarse_proto], dim=1)
            base_data.append(idx_weight)
        base_proto = torch.cat([pos for pos in base_data], dim=0)
        return base_proto

    def extra_coarse_weight(self, belong_cluster):
        belong_coarse = []

        for i in range(len(belong_cluster)):
            coarse_proto = self.fc_coarse.weight[belong_cluster[i], :].contiguous().view(1, -1, 640)
            belong_coarse.append(coarse_proto)
        coarse_protos = torch.cat([neg for neg in belong_coarse], dim=0)
        return coarse_protos

    def find_continue_topk(self, score, k, label=None, testing=False):
        coarse_matrix = torch.zeros([self.args.way, len(self.coarse)]).cuda()
        coarse_label = torch.zeros([self.args.way]).long().cuda()
        indexs_topk_matrix = torch.zeros([self.args.way, k]).long().cuda()
        list_way = []
        for i in range(self.args.way):
            ranking = []
            sort_ranking = []
            coarse = copy.deepcopy(self.coarse)
            for j in range(len(self.coarse)):
                tmp_idx = coarse[j]

                if (not testing) and (label[i] in tmp_idx):
                    tmp_idx.remove(label[i])
                idx_tensor = torch.tensor(tmp_idx)
                idx_score = score[i, idx_tensor].contiguous().view(-1)
                idx_score_sorted, indexs = torch.sort(idx_score, descending=True)
                if k < len(tmp_idx):
                    coarse_topk = idx_score_sorted[:k]
                else:
                    coarse_topk = idx_score_sorted
                coarse_matrix[i, j] = torch.mean(coarse_topk)
                index_to_label = idx_tensor[indexs]
                ranking.append(index_to_label)
            cluster_sorted, cluster_indexs_sorted = torch.sort(coarse_matrix[i, :], descending=True)
            coarse_label[i] = cluster_indexs_sorted[0]
            for idx in range(len(self.coarse)):
                for item in ranking[cluster_indexs_sorted[idx]]:
                    sort_ranking.append(item)
            sort_ranking = torch.tensor(sort_ranking)
            list_way.append(sort_ranking)

        idx_matrixs = torch.stack([topk for topk in list_way], dim=0)
        indexs_topk_matrix[:, :] = idx_matrixs[:, :k]
        return coarse_label, indexs_topk_matrix
