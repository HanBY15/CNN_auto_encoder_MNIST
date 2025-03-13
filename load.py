import model
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='cnnAE')
parser.add_argument('-lr','--learning_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('--seed',type=int, help='random seed', default=0)
parser.add_argument('--batch_size', type=int, default=256)

PATH = 'state_dict_model.pth'

if __name__=='__main__':
    args = parser.parse_args()
    net = model.cnnAE(3, args.seed, args.learning_rate)
    net.load_state_dict(torch.load(PATH))