import torch
import os
def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'

def add_noise(tensor, mean=0, std=1):
    noise = torch.rand(tensor.size())*std+mean
    noisy_tensor = tensor+noise
    return noisy_tensor

def make_dir():
    file_dir = ''
    if not os.path.exists(file_dir):
        os.makefirs(file_dir)

def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 32, 32)
    save_image(img, name)