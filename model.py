import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import utils 

class cnnAE(nn.Module):
    def __init__(self, encoded_dim, seed, lr):
        super(cnnAE, self).__init__()
        self.lr = lr  #学习率
        torch.manual_seed(seed)
        self.device = utils.get_device()
        self.loss_fn = torch.nn.MSELoss()  #损失函数
        self.encoder = Encoder(encoded_space_dim=encoded_dim)
        self.encoder.to(self.device)
        self.decoder = Decoder(encoded_space_dim=encoded_dim)
        self.decoder.to(self.device)
        self.params_to_optimize = [
            {'params':self.encoder.parameters()},
            {'params':self.decoder.parameters()}
        ]
        self.optim = torch.optim.Adam(self.params_to_optimize, lr=self.lr, weight_decay=1e-05)
    def train_epoch_den(self, dataloader, noise_factor=0.3):
        self.encoder.train()
        self.decoder.train()
        train_loss = []
        # Iterate the dataloader
        for image_batch, _ in dataloader:
            image_noisy = utils.add_noise(image_batch, std=noise_factor)
            image_noisy = image_noisy.to(self.device)
            encoded_data = self.encoder(image_noisy)
            decoded_data = self.decoder(encoded_data)
            loss = self.loss_fn(decoded_data, image_noisy)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            print(f'partial train loss (single batch):{loss.data}')
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)
    def test_epoch(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            conc_out = []
            conc_label = []
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)
                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())
            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = self.loss_fn(conc_out, conc_label)
        return val_loss.data
    def save_para(self, path):
        torch.save(self.state_dict(), path)

    # def load_para(self, path):
    #     self.load_state_dict(path)
    #     self.eval()
        

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x