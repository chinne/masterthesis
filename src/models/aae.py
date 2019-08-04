import argparse
import itertools
import math
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from ..common import accuracy_XGboost

Tensor = torch.FloatTensor


def reparameterization(mu, logvar, z_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), z_dim))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self, data_dim, h_dim, z_dim):
        self.z_dim = z_dim
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(data_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, data):
        x = self.model(data)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.z_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, data_dim, h_dim, z_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, data_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, z_dim, h_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, h_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim * 2, h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(
        gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True
    )


# ----------
#  Training
# ----------
def train(
    dataloader, prefix:str, data_dim: int, h_dim: int, z_dim: int, lr:float, num_epochs:int, feature_cols, label_col=[]):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()
    # Initialize generator and discriminator
    encoder = Encoder(data_dim, h_dim, z_dim)
    decoder = Decoder(data_dim, h_dim, z_dim)
    discriminator = Discriminator(z_dim, h_dim)

    #     if cuda:
    #         encoder.cuda()
    #         decoder.cuda()
    #         discriminator.cuda()
    #         adversarial_loss.cuda()
    #         pixelwise_loss.cuda()

    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()),
        lr=lr,
        betas=(0.5, 0.9),
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    Tensor = torch.FloatTensor  # torch.cuda.FloatTensor if cuda else torch.FloatTensor
    G_losses = []
    D_losses = []
    xgb_losses = []

    print("Starting Training Loop...")
    start_time = time()

    for epoch in range(num_epochs):
        G_losses_iter = []
        D_losses_iter = []
        generated_data = []
        real_data_list = []
        for i, data in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(data.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(data.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            #             real_imgs = Variable(imgs.type(Tensor))
            real_data = Variable(data.type(Tensor))
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            #             encoded_imgs = encoder(real_imgs)
            encoded_data = encoder(real_data)
            #             decoded_imgs = decoder(encoded_imgs)
            decoded_data = decoder(encoded_data)

            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adversarial_loss(
                discriminator(encoded_data), valid
            ) + 0.999 * pixelwise_loss(decoded_data, real_data)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], z_dim))))

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(z), valid)
            fake_loss = adversarial_loss(discriminator(encoded_data.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

  

            batches_done = epoch * len(dataloader) + i
            generated_data.extend(decoded_data.detach().numpy())
            real_data_list.extend(data.numpy())

            G_losses_iter.append(g_loss.item())
            D_losses_iter.append(d_loss.item())
        G_losses_iter_mean = sum(G_losses_iter) / len(G_losses_iter)
        D_losses_iter_mean = sum(D_losses_iter) / len(D_losses_iter)
        G_losses.append(G_losses_iter_mean)
        D_losses.append(D_losses_iter_mean)
        if epoch % 10 is 0:
            xgb_loss = accuracy_XGboost.CheckAccuracy(real_data_list, generated_data, feature_cols, label_col)
            xgb_losses = np.append(xgb_losses, xgb_loss)
            print(f"epoch: {epoch}, Accuracy: {xgb_loss}")
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
            torch.save({
                "Epochs": epoch,
                "decoder": decoder.state_dict()
                }, "{}/decoder_{}_{}.pth".format('models', prefix, epoch))
    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return xgb_losses, G_losses, D_losses


def generate_data(epoch, data_dim: int, h_dim: int, z_dim: int, amount, device):
    if device is 'cpu':
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor
    decoder = Decoder(data_dim, h_dim, z_dim)
    checkpoint = torch.load(f'models/decoder_{str(epoch)}.pth')
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    noise = Variable(Tensor(np.random.normal(0, 1, (amount, z_dim))))
    generated_data = decoder(noise)
    generated_data = torch.tanh(generated_data)
    return generated_data


# def generate(self, n):
#         data_dim = self.transformer.output_dim
#         decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)

#         ret = []
#         for epoch in self.store_epoch:
#             checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
#             decoder.load_state_dict(checkpoint['decoder'])
#             decoder.eval()
#             decoder.to(self.device)

#             steps = n // self.batch_size + 1
#             data = []
#             for i in range(steps):
#                 mean = torch.zeros(self.batch_size, self.embeddingDim)
#                 std = mean + 1
#                 noise = torch.normal(mean=mean, std=std).to(self.device)
#                 fake, sigmas = decoder(noise)
#                 fake = torch.tanh(fake)
#                 data.append(fake.detach().cpu().numpy())
#             data = np.concatenate(data, axis=0)
#             data = data[:n]
#             data = self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
#             ret.append((epoch, data))
#         return ret

#             if batches_done % opt.sample_interval == 0:
#                 sample_image(n_row=10, batches_done=batches_done)