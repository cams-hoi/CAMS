import os
import math
import numpy as np
import torch
import torch.distributed as dist

def dist_init():

    rank = int(os.environ["RANK"])
    # rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend='nccl')
    dist.barrier()

    torch.cuda.set_device(rank)

    return rank, world_size

class DistributedPerEpochSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, world_size=None, shuffle_strategy=0, train=True):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        assert rank < world_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle_strategy = shuffle_strategy
        self.train = train

        self.epoch_len = int(math.ceil(len(self.dataset) / self.batch_size / self.world_size)) * self.batch_size
        self.total_size = self.epoch_len * self.world_size

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.epoch_len

    def gen_list(self):

        if self.shuffle_strategy == 0:

            indices = list(range(len(self.dataset)))
            indices += [indices[-1]] * (self.total_size - len(indices))
            indices = indices[self.rank*self.epoch_len:(self.rank+1)*self.epoch_len]

            if self.train:
                np.random.shuffle(indices)

        return indices

    def set_epoch(self, epoch=0):

        np.random.seed(epoch)
        self.indices = self.gen_list()
        self.dataset.indices = self.indices
