import os
import argparse
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

from core.utils.dist_utils import dist_init
from core.utils.util import set_randomseed, get_auto_resume
from core.config import Config
from core.solvers import solver_entry

def main():
    args = parser.parse_args()
    args.load_path = get_auto_resume(os.path.dirname(args.config))

    C = Config(args.config)
    if C.config.get('seed', None) is not None:
        set_randomseed(C.config['seed'])

    rank, world_size = dist_init()
    C.config['rank'] = rank
    C.config['world_size'] = world_size

    S = solver_entry(C)
    S.initialize(args)
    if not args.inference:
        S.run()
    else:
        S.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pretrain")

    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--load-path', type=str, default='')
    parser.add_argument('--ignore', nargs='+', default=[], type=str)
    parser.add_argument('--inference', action='store_true', default=False)

    main()
