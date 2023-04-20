import time
import os.path as osp
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from trainer.base import Trainer
from trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
from logger import Mylog, set_log_path, log
from sklearn.metrics import roc_auc_score

class MGCTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.set_num = args.way * args.query
        if args.way != args.eval_way:
            self.set_num = args.eval_way * args.query
        self.open_num = args.eval_open_way * args.eval_query

        self.model, self.para_model = prepare_model(args)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def auroc(self, pos_scores, neg_scores):
        y_true = np.array([1] * self.set_num + [0] * self.open_num)
        known_scores = pos_scores.detach().cpu().numpy()
        unknown_scores = neg_scores.detach().cpu().numpy()
        y_score = np.concatenate([known_scores, unknown_scores])
        auc_score = roc_auc_score(y_true, y_score)
        return auc_score
    def auroc_bcls(self, neg_scores):
        y_true = np.array([0] * self.open_num + [1] * self.set_num)
        auc_score = roc_auc_score(y_true, neg_scores)
        return auc_score
    def prepare_label(self):
        args = self.args
        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_open = torch.arange(args.way, args.way + 1, dtype=torch.int16).repeat(args.way * args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        label_all = torch.cat([label, label_open])

        label = label.type(torch.LongTensor)
        label_all = label_all.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_all = label_all.cuda()
            label_aux = label_aux.cuda()
            label = label.cuda()

        return label, label_all, label_aux

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        label, label_all, label_aux = self.prepare_label()
        bcls_label = np.array([0] * self.set_num + [1] * self.open_num)
        bcls_label = torch.from_numpy(bcls_label).type(torch.FloatTensor)

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl1 = Averager()
            tl2 = Averager()
            tl3 = Averager()
            ta = Averager()
            tauroc = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1
                if torch.cuda.is_available():
                    data, gt_label, coarse_label = [_.cuda() for _ in batch]

                else:
                    data, gt_label, coarse_label = batch[0], batch[1], batch[2]
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                logits, reg_logits, cls_logits, bcls_score, plain_proto, enhance_proto, instance_logits = self.model(data, gt_label)
                if args.naive:
                    pos_query, _ = logits.split([self.set_num, self.open_num], dim=0)
                else:
                    pos_query, _ = instance_logits.split([self.set_num, self.open_num], dim=0)

                loss_cls = F.cross_entropy(pos_query, label)

                total_loss = loss_cls


                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(pos_query, label)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            log('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
                , filename='trainfsl.txt')

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')


    def evaluate_test(self, name='max_auroc_instance.pth', filename='trainfsl.txt'):
        # restore model args
        args = self.args
        args.way = args.eval_way
        set_num = args.way * args.query
        log(osp.join(self.args.save_path, name), filename=filename)
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, name))['params'])
        self.model.eval()
        record = np.zeros((args.test_episodes, 10))  # loss and acc and auroc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label_open = torch.arange(args.way, args.way + 1, dtype=torch.int16).repeat(args.way * args.query)
        label_all = torch.cat([label, label_open])
        label = label.type(torch.LongTensor)
        label_all = label_all.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits, reg_logits, cls_logits, bcls_score, plain_proto, enhance_proto, instance_logits = self.model(data, gt_label, testing=True)
                if args.source_logits:
                    logits = logits * args.temperature

                pos_query, neg_query = logits.split([self.set_num, self.open_num], dim=0)
                acc = count_acc(pos_query, label)
                instance_logits_pos, _ = instance_logits.split([self.set_num, self.open_num], dim=0)
                acc1 = count_acc(instance_logits_pos, label)

                if args.naive:
                    loss = F.cross_entropy(pos_query, label)
                else:
                    loss = F.cross_entropy(instance_logits_pos, label)


                if args.open_loss_type == 'MaxProbability':
                    if not args.test_logits:
                        logits = torch.softmax(logits, dim=1)
                    pos_logits = logits[:set_num]
                    neg_logits = logits[set_num:]
                    pos_max, _ = torch.max(pos_logits, dim=1, keepdim=True)
                    neg_max, _ = torch.max(neg_logits, dim=1, keepdim=True)
                    test_auroc = self.auroc(pos_max, neg_max)

                    if reg_logits is not None:
                        instance_pos_logits = instance_logits[:set_num]
                        instance_neg_logits = instance_logits[set_num:]
                        instance_pos_max, _ = torch.max(instance_pos_logits, dim=1, keepdim=True)
                        instance_neg_max, _ = torch.max(instance_neg_logits, dim=1, keepdim=True)
                        test_auroc_instance = self.auroc(instance_pos_max, instance_neg_max)
                else:
                    test_scores = bcls_score.detach().cpu().numpy()
                    test_auroc = self.auroc_bcls(test_scores)

                # auroc max
                pos_max_sim, _ = torch.max(pos_query, dim=1)
                neg_max_sim, _ = torch.max(neg_query, dim=1)
                logits_neg_pos = torch.cat([neg_max_sim, pos_max_sim], dim=0)
                logits_neg_pos = logits_neg_pos.detach().cpu().numpy()
                test_auroc_max = self.auroc_bcls(logits_neg_pos)


                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc
                record[i - 1, 2] = test_auroc
                record[i - 1, 3] = test_auroc_max
                record[i - 1, 4] = acc1
                record[i - 1, 5] = test_auroc_instance
        assert (i == record.shape[0])
        tl, _ = compute_confidence_interval(record[:, 0])
        ta, tap = compute_confidence_interval(record[:, 1])
        testauroc, testauroc_p = compute_confidence_interval(record[:, 2])
        testauroc_max, testauroc_max_p = compute_confidence_interval(record[:, 3])

        ta1, tap1 = compute_confidence_interval(record[:, 4])
        testauroc_instance, testauroc_p_instance = compute_confidence_interval(record[:, 5])

        self.trlog['test_acc'] = ta
        self.trlog['test_acc_interval'] = tap
        self.trlog['test_acc1'] = ta1
        self.trlog['test_acc1_interval'] = tap1
        self.trlog['test_loss'] = tl
        self.trlog['test_auroc'] = testauroc
        self.trlog['testauroc_p'] = testauroc_p
        self.trlog['testauroc_max'] = testauroc_max
        self.trlog['testauroc_max_p'] = testauroc_max_p

        self.trlog['test_auroc_instance'] = testauroc_instance
        self.trlog['testauroc_p_instance'] = testauroc_p_instance

        log('Test acc={:.4f} + {:.4f}, acc1={:.4f} + {:.4f}, test auroc={:.4f} + {:.4f}, test auroc_max={:.4f} + {:.4f}, test auroc_buquan={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval'],
            self.trlog['test_acc1'],
            self.trlog['test_acc1_interval'],
            self.trlog['test_auroc'],
            self.trlog['testauroc_p'],
            self.trlog['testauroc_max'],
            self.trlog['testauroc_max_p'],
            self.trlog['test_auroc_instance'],
            self.trlog['testauroc_p_instance']), filename=filename)

        return tl, ta, tap



