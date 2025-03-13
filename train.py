import argparse
import os
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import model

parser = argparse.ArgumentParser(description='cnnAE')
parser.add_argument('-lr','--learning_rate', type=float, help='learning rate', default=0.001)
parser.add_argument('--seed',type=int, help='random seed', default=0)
parser.add_argument('--batch_size', type=int, default=256)

PATH = 'state_dict_model.pth'
data_dir='.'
if __name__=='__main__':
    args = parser.parse_args()
    # print(args.learning_rate)
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor(),])
    test_transform = transforms.Compose([transforms.ToTensor(),])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_data, val_data = random_split(train_dataset, [int(len(train_dataset)-len(train_dataset)*0.2), int(len(train_dataset)*0.2)])
    batch_size = args.batch_size
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    net = model.cnnAE(3, args.seed, args.learning_rate)   
    # print(f'net average MSE loss: {net.test_epoch(test_loader)}')
    if os.path.exists(PATH):
        net.load_state_dict(torch.load(PATH))
    net.train_epoch_den(train_loader)
    net.save_para(path=PATH)
    print(f'net average MSE loss: {net.test_epoch(test_loader)}')
