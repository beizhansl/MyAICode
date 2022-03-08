import os

import argparse

from torch.backends import cudnn
from config import config, merge_cfg_arg

from data_loaders.dataloder import get_loader
from solver_scfe import Solver_SCFEGAN

path = 'C:\\Users\\zhans\\Desktop\\dataset'

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')
    # general
    parser.add_argument('--data_path', default=path, type=str, help='training and test data path')
    parser.add_argument('--batch_size', default='1', type=int, help='batch_size')
    parser.add_argument('--vis_step', default='1000', type=int, help='steps between visualization')
    parser.add_argument('--task_name', default='train', type=str, help='task name')
    parser.add_argument('--checkpoint', default='64_4000', type=str, help='checkpoint to load')
    parser.add_argument('--ndis', default='2', type=int, help='train discriminator steps')
    parser.add_argument('--LR', default="5e-4", type=float, help='Learning rate')
    parser.add_argument('--decay', default='100', type=int, help='epochs number for training')
    parser.add_argument('--epochs', default='200', type=int, help='nums of epochs')
    args = parser.parse_args()
    return args
#
def train_net():
    # enable cudnn
    cudnn.benchmark = True
    data_loaders = get_loader(config, mode="train")    # return train&test
    #get the solver
    solver = Solver_SCFEGAN(data_loaders, config)
    solver.train()

if __name__ == '__main__':
    args = parse_args()
    config = merge_cfg_arg(config, args)

    print('           ⊱ ──────ஓ๑♡๑ஓ ────── ⊰')
    print('🎵 hhey, arguments are here if you need to check 🎵')
    for arg in vars(config):
        print('{:>15}: {:>30}'.format(str(arg), str(getattr(config, arg))))
    print()
    # Create the directories if not exist
    if not os.path.exists(config.data_path):
        print("No datapath!!", config.data_path)
        exit()

    train_net()