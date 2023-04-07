import numpy as np


def default(init_lr, global_step):
    return init_lr

# https://github.com/tensorflow/tensor2tensor/issues/280#issuecomment-339110329
class noam_learning_rate_decay(object):

    def __init__(self, init_lr, optimizer, warmup_steps=4000, minimum=None):
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.minimum = minimum
        self.optimizer = optimizer

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, global_step):
         # Noam scheme from tensor2tensor:
        self.warmup_steps = float(self.warmup_steps)
        step = global_step + 1.
        lr = self.init_lr * self.warmup_steps**0.5 * np.minimum(
            step * self.warmup_steps**-1.5, step**-0.5)
        if self.minimum is not None and global_step > self.warmup_steps:
            if lr < self.minimum:
                lr = self.minimum

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def step_learning_rate_decay(init_lr, global_step,
                             anneal_rate=0.98,
                             anneal_interval=30000):
    return init_lr * anneal_rate ** (global_step // anneal_interval)


def cyclic_cosine_annealing(init_lr, global_step, T, M):
    """Cyclic cosine annealing

    https://arxiv.org/pdf/1704.00109.pdf

    Args:
        init_lr (float): Initial learning rate
        global_step (int): Current iteration number
        T (int): Total iteration number (i,e. nepoch)
        M (int): Number of ensembles we want

    Returns:
        float: Annealed learning rate
    """
    TdivM = T // M
    return init_lr / 2.0 * (np.cos(np.pi * ((global_step - 1) % TdivM) / TdivM) + 1.0)
