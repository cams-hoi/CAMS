import os
import re
import copy
import random
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.backends import cudnn

class kld_weight_scheduler(object):

    def __init__(self, config=None):

        self.config = config
        if self.config is not None:

            self.kld_max = self.config.kld_max
            self.kld_warmup_iters = self.config.kld_warmup_iters
            self.kld_start_iter = self.config.kld_start_iter

            self.kld_inc = (self.kld_max - 0) / self.kld_warmup_iters

    def get(self, iter):

        if self.config is None:
            kld_weight = 1
        else:
            if iter < self.kld_start_iter:
                kld_weight = 0
            elif iter < self.kld_start_iter + self.kld_warmup_iters:
                kld_weight = self.kld_inc * (iter - self.kld_start_iter)
            else:
                kld_weight = self.kld_max

        return kld_weight

def get_auto_resume(log_dir):

    max_iter = 0
    load_path = os.path.join(log_dir, 'checkpoints')
    if os.path.exists(load_path):
        files = os.listdir(load_path)
        for file in files:
            if file.startswith('ckpt_epoch') and file.endswith('.pth.tar'):
                cur_iter = int(file.split('iter')[1].split('.')[0])
                if cur_iter > max_iter:
                    max_iter = cur_iter
                    max_file = file

    if max_iter == 0:
        load_path = None
    else:
        print('resume from iter {}'.format(str(max_iter)))
        load_path = os.path.join(load_path, max_file)

    return load_path

def load_state(load_path, model, optimizer=None, rank=0, ignore=[], replace=None):

    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location='cpu')

        if len(ignore) > 0:

            for k in list(checkpoint['state_dict'].keys()):
                flag = False
                for prefix in ignore:
                     if k.startswith(prefix):
                         flag = True
                         the_prefix = prefix
                         break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint['state_dict'][k]

        if replace is not None:

            for k in list(checkpoint['state_dict'].keys()):

                if k.startswith(replace):
                    checkpoint['state_dict'][k.replace(replace, '')] = checkpoint['state_dict'][k]
                    del checkpoint['state_dict'][k]

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if getattr(model, 'set_actnorm_init', None) is not None:
            model.set_actnorm_init(inited=True)
        dist.barrier()

        if rank == 0:
            keys1 = set(checkpoint['state_dict'].keys())
            keys2 = set([k for k,_ in model.named_parameters()])
            not_loaded = keys2 - keys1
            for k in not_loaded:
                print('caution: {} not loaded'.format(k))

        return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(load_path)

# load model params before creating optimizer
def load_state_single(load_path, model, ignore=[]):

    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location='cpu')

        if len(ignore) > 0:

            for k in list(checkpoint['state_dict'].keys()):
                flag = False
                for prefix in ignore:
                     if k.startswith(prefix):
                         flag = True
                         the_prefix = prefix
                         break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint['state_dict'][k]

        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith('module.'):
                checkpoint['state_dict'][k.replace('module.', '')] = checkpoint['state_dict'][k]
                del checkpoint['state_dict'][k]

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if getattr(model, 'set_actnorm_init', None) is not None:
            model.set_actnorm_init(inited=True)

        keys1 = set(checkpoint['state_dict'].keys())
        keys2 = set([k for k,_ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print('caution: {} not loaded'.format(k))

        return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(load_path)

def save_state(epoch, iter, model, optimizer, log_dir, is_best=False, max_saving_num=10):

    state = {
        'step': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, 'ckpt_epoch{}_iter{}.pth.tar'.format(epoch, iter))
    best_path = os.path.join(ckpt_dir, 'ckpt_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        torch.save(state, best_path)
    if max_saving_num:
        ckpts = [x for x in os.listdir(ckpt_dir) if x.startswith('ckpt_epoch')]
        iters = [int(x.split('.pth')[0].split('iter')[1]) for x in ckpts]
        if len(ckpts) >= max_saving_num:
            history_ckpt = os.path.join(ckpt_dir, ckpts[np.argmin(iters)])
            os.remove(history_ckpt)

def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)20s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.propagate = 0
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def set_randomseed(seed):
    cudnn.deterministic = True
    cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_proper_cuda_device(device, verbose=True):
    if not isinstance(device, list):
        device = [device]
    count = torch.cuda.device_count()
    if verbose:
        print("[Builder]: Found {} gpu".format(count))
    for i in range(len(device)):
        d = device[i]
        did = None
        if isinstance(d, str):
            if re.search("cuda:[\d]+", d):
                did = int(d[5:])
        elif isinstance(d, int):
            did = d
        if did is None:
            raise ValueError("[Builder]: Wrong cuda id {}".format(d))
        if did < 0 or did >= count:
            if verbose:
                print("[Builder]: {} is not found, ignore.".format(d))
            device[i] = None
        else:
            device[i] = did
    device = [d for d in device if d is not None]
    return device

def get_proper_device(devices, verbose=True):
    origin = copy.copy(devices)
    devices = copy.copy(devices)
    if not isinstance(devices, list):
        devices = [devices]
    use_cpu = any([d.find("cpu")>=0 for d in devices])
    use_gpu = any([(d.find("cuda")>=0 or isinstance(d, int)) for d in devices])
    assert not (use_cpu and use_gpu), "{} contains cpu and cuda device.".format(devices)
    if use_gpu:
        devices = get_proper_cuda_device(devices, verbose)
        if len(devices) == 0:
            if verbose:
                print("[Builder]: Failed to find any valid gpu in {}, use `cpu`.".format(origin))
            devices = ["cpu"]
    return devices
