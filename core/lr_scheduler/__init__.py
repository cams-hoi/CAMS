from .base import *

def lr_scheduler_entry(config):
    return globals()[config['type']](**config['kwargs'])
