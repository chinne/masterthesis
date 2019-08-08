import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.autograd import Variable

import torch.autograd as autograd
import random
from time import time


from ..common import accuracy_XGboost

randomSeed=42
random.seed(randomSeed)
torch.manual_seed(randomSeed)




class Generator(nn.Module):
    '''
    This is the generator of the GAN
    '''
    def __init__(self, randomNoise_dim, hidden_dim, realData_dim):
        '''
        Args:
            randomNoise_dim: An integer indicating the size of random noise input.
            hidden_dim: An integer indicating the size of the first hidden dimension.
            realData_dim: An integer indicating the real data dimension.
        '''
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(randomNoise_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, realData_dim)
        )

    def forward(self, input):
        return self.main(input)

class Critic(nn.Module):
    '''
    This is the Critic of the WGAN-GP
    '''
    def __init__(self, realData_dim, hidden_dim):
        '''
        Args:
            realData_dim: An integer indicating the real data dimension.
            hidden_dim: An integer indicating the size of the first hidden dimension.
        '''
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(realData_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        return self.main(input)


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return torch.mean(y_true * y_pred)


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))#, 1, 1)))
    # Get random interpolation between real and fake samples
# #     print(alpha.shape)

    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake  = Variable(Tensor(real_samples.size(0), 1).fill_(0.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Loss weight for gradient penalty
lambda_gp = 10

def train(dataloader, randomNoise_dim:int, hidden_dim: int, realData_dim:int, lr:float, num_epochs:int, feature_cols, label_col=[], device='cpu'):
    
    if device == 'cuda':
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor
        
        
    G_losses = []
    C_losses = []
    xgb_losses = []
    
    netG = Generator(randomNoise_dim, hidden_dim, realData_dim).to(device)
    netC = Critic(realData_dim, hidden_dim).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # set the train mode - Just necessary if dropout used
    # netG.train()
    # netD.train()    
    
    print("Starting Training Loop...")
    start_time = time()
    
    for epoch in range(num_epochs+1):
        G_losses_iter = []
        C_losses_iter = []
        generated_data = []
        real_data_list = []
        for i, data in enumerate(dataloader):

            valid = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)
            real_data = Variable(data.type(Tensor))

            # Train Discriminator on real data
            netC.zero_grad()

            noise = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], randomNoise_dim))))
            fake_samples = netG(noise)

            predC_real = netC(real_data)
            predC_fake = netC(fake_samples)

            gradient_penalty = compute_gradient_penalty(netC, real_data, fake_samples)

            errC_loss = -torch.mean(predC_real) + torch.mean(predC_fake) + lambda_gp * gradient_penalty

            errC_loss.backward()
            optimizerC.step()


            netG.zero_grad()
            if i % 5 == 0:
                fake_samples = netG(noise)
                predC_fake = netC(fake_samples)
                errG = -torch.mean(predC_fake)
                errG.backward()
                optimizerG.step()

            # Save Losses for plotting later
            generated_data.extend(fake_samples.detach().numpy())
            real_data_list.extend(real_data.numpy())
            
            G_losses_iter.append(errG.item())
            C_losses_iter.append(errC_loss.item())
        
        G_losses_iter_mean = sum(G_losses_iter)/len(G_losses_iter)
        C_losses_iter_mean = sum(C_losses_iter)/len(C_losses_iter)
        G_losses.append(G_losses_iter_mean)
        C_losses.append(C_losses_iter_mean)

        
        if epoch % 10 is 0:
            xgb_loss = accuracy_XGboost.CheckAccuracy(real_data_list, generated_data, feature_cols)
            xgb_losses = np.append(xgb_losses, xgb_loss)
            print(f'epoch: {epoch}, Accuracy: {xgb_loss}')
            print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f\t'
                    % (epoch, num_epochs+1, i, len(dataloader),
                        errC_loss.item(), errG.item()))
            torch.save({
                    "Epochs": epoch,
                    "generator": netG.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                }, "models/wgangp/generator/generator_{}.pth".format(epoch))
            
            torch.save({
                    "Epochs": epoch,
                    "critic": netC.state_dict(),
                    "optimizerD": optimizerC.state_dict()
                }, "models/wgangp/critic/critic_{}.pth".format(epoch))
            # # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            accuracy_XGboost.PlotData(real_data_list, generated_data, feature_cols, label_col, seed=42, data_dim=2)
            


    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return xgb_losses, G_losses, C_losses

def generate_data(epoch: int, randomNoise_dim: int, hidden_dim: int, realData_dim: int, amount: int, device: str):
    if device is 'cpu':
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor
    netG = Generator(randomNoise_dim, hidden_dim, realData_dim).to(device)
    checkpoint = torch.load(f'models/wgangp/generator/generator_{str(epoch)}.pth')
    netG.load_state_dict(checkpoint['generator'])
    noise = Variable(Tensor(np.random.normal(0, 1, (amount, randomNoise_dim))))
    generated_data = netG(noise)
    return generated_data
