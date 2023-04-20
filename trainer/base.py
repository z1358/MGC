import abc
import torch
import os.path as osp

from utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from logger import Logger
from logger import Mylog, set_log_path, log

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0
        self.trlog['max_acc_auroc'] = 0.0

        self.trlog['max_auroc'] = 0.0
        self.trlog['max_auroc_epoch'] = 0
        self.trlog['max_auroc_acc'] = 0.0
        self.trlog['max_auroc_acc_interval'] = 0.0

        self.trlog['max_acc1'] = 0.0
        self.trlog['max_acc1_epoch'] = 0
        self.trlog['max_acc1_interval'] = 0.0
        self.trlog['max_acc1_auroc'] = 0.0

        self.trlog['max_auroc_instance'] = 0.0
        self.trlog['max_auroc_epoch_instance'] = 0
        self.trlog['max_auroc_acc_instance'] = 0.0
        self.trlog['max_auroc_acc_interval_instance'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_test(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        args = self.args
        if self.train_epoch % args.eval_interval == 0:
            vl, va, vap, vauroc, va1, vap1, vauroc_instance= self.evaluate(self.val_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_epoch)
            self.logger.add_scalar('val_acc', float(va),  self.train_epoch)
            log('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f} auroc={:.4f} acc1={:.4f}+{:.4f} auroc_instance={:.4f}'.format(epoch, vl, va, vap, vauroc, va1, vap1, vauroc_instance), filename='trainfsl.txt')

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.trlog['max_acc_auroc'] = vauroc
                self.save_model('max_acc')
            if vauroc >= self.trlog['max_auroc']:
                self.trlog['max_auroc_acc'] = va
                self.trlog['max_auroc_acc_interval'] = vap
                self.trlog['max_auroc_epoch'] = self.train_epoch
                self.trlog['max_auroc'] = vauroc
                self.save_model('max_auroc')

            if va1 >= self.trlog['max_acc1']:
                self.trlog['max_acc1'] = va1
                self.trlog['max_acc1_interval'] = vap1
                self.trlog['max_acc1_epoch'] = self.train_epoch
                self.trlog['max_acc1_auroc'] = vauroc_instance
                self.save_model('max_acc1')

            if vauroc_instance >= self.trlog['max_auroc_instance']:
                self.trlog['max_auroc_acc_instance'] = va
                self.trlog['max_auroc_acc_interval_instance'] = vap
                self.trlog['max_auroc_epoch_instance'] = self.train_epoch
                self.trlog['max_auroc_instance'] = vauroc_instance
                self.save_model('max_auroc_instance')

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('grad_norm',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
