import os
import time
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from manotorch.manolayer import ManoLayer

from core.datasets import dataset_entry
from core.models import model_entry
from core.lr_scheduler import lr_scheduler_entry
from core.utils.util import load_state, save_state, create_logger, kld_weight_scheduler
from core.utils.dist_utils import DistributedPerEpochSampler
from core.loss.stagecvae_all_loss import *

class Solver(object):

    def __init__(self, C):

        self.config = edict(C.config)
        self.rank = self.config.rank
        self.world_size = self.config.world_size
        self.last_epoch = -1

        # for logs, should only be in rank0
        save_path = self.config.log_dir
        if self.rank == 0:
            if not os.path.exists('{}/events'.format(save_path)):
                os.makedirs('{}/events'.format(save_path))
            if not os.path.exists('{}/logs'.format(save_path)):
                os.makedirs('{}/logs'.format(save_path))
            if not os.path.exists('{}/checkpoints'.format(save_path)):
                os.makedirs('{}/checkpoints'.format(save_path))
            self.tb_logger = SummaryWriter('{}/events'.format(save_path))
            self.logger = create_logger('global_logger', '{}/logs/log.txt'.format(save_path))
        dist.barrier()

        dir_weight = self.config.get('dir_weight', 100)
        joints_weight = self.config.get('joints_weight', 1)
        self.cvae_all_loss = CVAEALLLoss(dir_weight, joints_weight)

    def initialize(self, args):

        self.create_dataset()
        self.create_model()
        self.create_optimizer()
        self.load(args)
        self.create_dataloader()
        self.create_lr_scheduler()

        if self.rank == 0:
            self.logger.info(self.config)

    def create_dataset(self):
        self.train_dataset = dataset_entry(self.config)
        self.val_dataset = self.train_dataset.get_test_dataset()

    def create_model(self):

        self.model = model_entry(self.config.model)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank], output_device=self.rank)

    def create_optimizer(self):

        if self.config.optim.type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), **self.config.optim.kwargs)
        elif self.config.optim.type == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.optim.kwargs.lr)
        else:
            raise ValueError("NotImplemented Optimizer: {}".format(self.config.optim.type))

    def create_dataloader(self):

        self.train_sampler = DistributedPerEpochSampler(self.train_dataset,
                                                    self.config.batch_size,
                                                    self.rank,
                                                    self.world_size,
                                                    shuffle_strategy=0,
                                                    train=True)

        self.val_sampler = DistributedPerEpochSampler(self.val_dataset,
                                                    self.config.batch_size,
                                                    self.rank,
                                                    self.world_size,
                                                    shuffle_strategy=0,
                                                    train=False)

        self.dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.config.batch_size,
                                      sampler=self.train_sampler,
                                      num_workers=0)

        self.val_dataloader = DataLoader(self.val_dataset,
                                      batch_size=self.config.batch_size,
                                      sampler=self.val_sampler,
                                      num_workers=0)

        ## random seed setting
        self.max_iter = self.config.max_epoch * len(self.dataloader)
        rng = np.random.RandomState(self.config.get('seed', 0))
        self.randomseed_pool = rng.randint(999999, size=self.max_iter)

    def create_lr_scheduler(self):

        self.config.lr_scheduler.kwargs.init_lr = self.config.optim.kwargs.lr
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer
        self.lr_scheduler = lr_scheduler_entry(self.config.lr_scheduler)

    def load(self, args):
        if args.load_path is not None:
            self.last_epoch = load_state(args.load_path, self.model, self.optimizer, self.rank, args.ignore)

    def save(self, is_best=False):
        save_state(self.current_epoch, self.current_iter, self.model,
                            self.optimizer, self.config.log_dir, is_best)

    def forward(self):

        ## set random seed with current_iter at each iteration
        self._set_randomseed(self.randomseed_pool[self.current_iter])

        # forward phase
        y = self.model(self.input)

        self.binary_loss, self.pos_L2_loss, \
        self.dir_L2_loss, self.KLD_loss, \
        self.seq_tip_loss, self.seq_joints_loss = self.cvae_all_loss(y)

        self.loss = self.binary_loss + self.pos_L2_loss + self.dir_L2_loss + \
                    self.KLD_loss + self.seq_tip_loss + self.seq_joints_loss

        if self.rank == 0:
            save_path = os.path.join(self.config.log_dir, 'tmp/train')
            if isinstance(y, dict):
                torch.save(y, os.path.join(save_path, 'output'))
            else:
                torch.save(y.data.cpu(), os.path.join(save_path, 'output'))

        dist.barrier()

    def backward(self):

        # self.model.zero_grad()
        self.optimizer.zero_grad()
        self.loss.backward()

    def validate(self):

        # Validation forward phase
        loss_val = 0
        min_loss_val = 1000000
        n_batches = 0
        save_dict = {}
        # with torch.no_grad():

        self.val_sampler.set_epoch()
        for ii, self.input in enumerate(self.val_dataloader):

            if self.rank == 0:
                save_path = os.path.join(self.config.log_dir, 'tmp/val')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.input, os.path.join(save_path, 'input_{}'.format(ii)))

            for k, v in self.input.items():
                if not isinstance(v, list):
                    self.input[k] = v.cuda()

            # forward phase
            with torch.no_grad():
                y = self.model.module.inference(self.input)

            save_dict[ii] = y

            n_batches = n_batches + 1

        if self.rank == 0:
            for ii, y in save_dict.items():
                if isinstance(y, dict):
                    torch.save(y, os.path.join(save_path, 'output_{}'.format(ii)))
                else:
                    torch.save(y.data.cpu(), os.path.join(save_path, 'output_{}'.format(ii)))

        if self.rank == 0:
            self.logger.info('Inference OK')

    def update(self):

        # need reduce in multi gpus
        # self.model.reduce_gradients()

        # operate grad
        if self.config.get('max_grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.config.max_grad_clip)
        if self.config.get('max_grad_norm', 0) > 0:
            self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()

    def run(self):

        end = time.time()
        for self.current_epoch in range(self.last_epoch+1, self.config.max_epoch):

            self.train_sampler.set_epoch(self.current_epoch)
            for i, self.input in enumerate(self.dataloader):
                # set to train state
                self.model.train()

                self.load_time = time.time() - end
                end = time.time()

                if self.rank == 0:
                    save_path = os.path.join(self.config.log_dir, 'tmp/train')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(self.input, os.path.join(save_path, 'input'))

                for k, v in self.input.items():
                    if not isinstance(v, list):
                        self.input[k] = v.cuda()

                self.current_iter = self.current_epoch * len(self.dataloader) + i
                self.lr_scheduler.step(self.current_iter)
                self.current_lr = self.lr_scheduler.get_lr()[0]

                self.forward()
                self.backward()
                self.update()

                self.iter_time = time.time() - end
                if self.current_iter % self.config.print_freq_iter == 0 and self.rank == 0:
                    self._tb_logging()
                    self._logging()

                # validation
                if self.current_iter % self.config.validation_log_gap == 0:
                    # set to eval state
                    self.model.eval()

                    self.validate()

                dist.barrier()

                end = time.time()

            # checkpoints
            if self.current_epoch % self.config.save_gap_epoch == 0 and self.current_epoch > 0 and self.rank == 0:
                self.save(is_best=True)
            dist.barrier()

        if self.rank == 0:
            self.tb_logger.close()

    def _set_randomseed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _tb_logging(self):

        self.tb_logger.add_scalar('lr', self.current_lr, self.current_iter)
        self.tb_logger.add_scalar('loss', self.loss, self.current_iter)
        self.tb_logger.add_scalar("grad_norm", self.grad_norm, self.current_iter)

    def _logging(self):

        self.logger.info('Epoch: [{}/{}]\t'
              'Iter: [{}/{}]\t'
              'Time {:.3f} (ETA:{:.2f}h) ({:.3f})\t'
              'Loss {:.4f}\t'
              'CFlag Loss {:.4f}\t'
              'Coord Loss {:.4f}\t'
              'Normal Loss {:.4f}\t'
              'KLD Loss {:.4f}\t'
              'Tip Loss {:.4f}\t'
              'Joints Loss {:.4f}\t'
              'Other Loss {:.4f}\t'
              'LR {:.8f}'.format(self.current_epoch, self.config.max_epoch, self.current_iter, self.max_iter,
              self.iter_time, self._ETA(), self.load_time, self.loss,self.binary_loss, self.pos_L2_loss,
              self.dir_L2_loss, self.KLD_loss, self.seq_tip_loss, self.seq_joints_loss, self.loss-self.dir_L2_loss,
              self.current_lr))

    def _ETA(self):
        return (self.iter_time + self.load_time) * (self.max_iter - self.current_iter) / 3600
