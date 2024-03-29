import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from torch.autograd import Variable

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

class Discriminator(nn.Module):
    '''
    This is the discriminator of the GAN
    '''
    def __init__(self, realData_dim, hidden_dim):
        '''
        Args:
            realData_dim: An integer indicating the real data dimension.
            hidden_dim: An integer indicating the size of the first hidden dimension.
        '''
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(realData_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



def train(dataloader, randomNoise_dim:int, hidden_dim: int, realData_dim:int, lr:float, num_epochs:int, device='cpu'):
    
    Tensor = torch.FloatTensor
#     else:
#         Tensor = torch.cuda.FloatTensor
        
        
    G_losses = []
    D_losses = []
    xgb_losses = []
    
    netG = Generator(randomNoise_dim, hidden_dim, realData_dim).to(device)
    netD = Discriminator(realData_dim, hidden_dim).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    # set the train mode - Just necessary if dropout used
    # netG.train()
    # netD.train()    
    num_epochs = num_epochs+1
    print("Starting Training Loop...")
    start_time = time()
    
    for epoch in range(num_epochs):
        G_losses_iter = []
        D_losses_iter = []
        generated_data = []
        real_data_list = []
        for i, data in enumerate(dataloader):

            valid = Variable(Tensor(data.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(Tensor(data.size(0), 1).fill_(0.0), requires_grad=False)
            real_data = Variable(data.type(Tensor))
            
            noise = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], randomNoise_dim))))

            optimizerG.zero_grad()
            fake_samples = netG(noise)
            #netG.zero_grad()
            predD_fake = netD(fake_samples)
            g_loss = adversarial_loss(predD_fake, valid)
            g_loss.backward()
            optimizerG.step()

            # Train Discriminator on real data
            optimizerD.zero_grad()
            #netD.zero_grad()
            predD_real = netD(real_data)
            real_loss = adversarial_loss(predD_real, valid)
            fake_loss = adversarial_loss(predD_fake, fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizerD.step()


            
    
            
              

            # Save Losses for plotting later
            generated_data.extend(fake_samples.detach().numpy())
            real_data_list.extend(real_data.numpy())
            
            G_losses_iter.append(g_loss.item())
            D_losses_iter.append(d_loss.item())
        
        G_losses_iter_mean = sum(G_losses_iter)/len(G_losses_iter)
        D_losses_iter_mean = sum(D_losses_iter)/len(D_losses_iter)
        G_losses.append(G_losses_iter_mean)
        D_losses.append(D_losses_iter_mean)

        
        if epoch % 10 is 0:
            xgb_loss = accuracy_XGboost.CheckAccuracy(real_data_list, generated_data)
            xgb_losses = np.append(xgb_losses, xgb_loss)
            print(f'epoch: {epoch}, Accuracy: {xgb_loss}')
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, num_epochs+1, i+1, len(dataloader),
                        g_loss.item(), d_loss.item()))
            torch.save({
                    "Epochs": epoch,
                    "generator": netG.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                }, "{}/generator_{}.pth".format('models', epoch))
            
            torch.save({
                    "Epochs": epoch,
                    "discriminator": netD.state_dict(),
                    "optimizerD": optimizerD.state_dict()
                }, "{}/discriminator_{}.pth".format('models', epoch))
            # # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            


    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return xgb_losses, G_losses, D_losses

def generate_data(epoch: int, randomNoise_dim: int, hidden_dim: int, realData_dim: int, amount: int, device: str):
    if device is 'cpu':
        Tensor = torch.FloatTensor
    else:
        Tensor = torch.cuda.FloatTensor
    netG = Generator(randomNoise_dim, hidden_dim, realData_dim).to(device)
    checkpoint = torch.load(f'models/generator_{str(epoch)}.pth')
    netG.load_state_dict(checkpoint['generator'])
    noise = Variable(Tensor(np.random.normal(0, 1, (amount, randomNoise_dim))))
    generated_data = netG(noise)
    return generated_data
