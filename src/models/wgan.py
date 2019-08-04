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
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*4, realData_dim),
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
            nn.Linear(realData_dim, hidden_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input):
        return self.main(input)



def train(dataloader, randomNoise_dim:int, hidden_dim: int, realData_dim:int, lr:float, num_epochs:int, feature_cols, label_col=[], device='cpu'):
    
    Tensor = torch.FloatTensor
#     else:
#         Tensor = torch.cuda.FloatTensor
        
        
    G_losses = []
    D_Losses = []
    xgb_losses = []
    
    netG = Generator(randomNoise_dim, hidden_dim, realData_dim).to(device)
    netD = Discriminator(realData_dim, hidden_dim).to(device)
    
    print(netG)
    print(netD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    # optimizerG = optim.Adam(netG.parameters(), lr=lr)
    # optimizerD = optim.Adam(netD.parameters(), lr=lr)
    clip_value = 0.01
    n_critic = 5
    # set the train mode - Just necessary if dropout used
    # netG.train()
    # netD.train()    
    print("Starting Training Loop...")
    start_time = time()
    num_epochs = num_epochs +1

    for epoch in range(num_epochs):
        G_losses_iter = []
        D_Losses_iter = []
        generated_data = []
        real_data_list = []
        for i, data in enumerate(dataloader):
            valid = Variable(Tensor(data.size(0), 1).fill_(0), requires_grad=False)
            fake  = Variable(Tensor(data.size(0), 1).fill_(1), requires_grad=False)
            for iter in range(n_critic):
                real_data = Variable(data.type(Tensor))              
                noise = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], randomNoise_dim)))) #Create noise in batch size
                fake_samples = netG(noise).detach()
                optimizerD.zero_grad()

                lossD = - torch.mean(netD(real_data)) + torch.mean(netD(fake_samples))

                lossD.backward()
                optimizerD.step()          
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            
            noise = Variable(Tensor(np.random.normal(0, 1, (data.shape[0], randomNoise_dim)))) #Create noise in batch size
                
            gen_samples = netG(noise)
            optimizerG.zero_grad()
            lossG = -torch.mean(netD(gen_samples))
            lossG.backward()
            optimizerG.step()

            

            # Save Losses for plotting later
            generated_data.extend(gen_samples.detach().numpy())
            real_data_list.extend(real_data.numpy())
            
            D_Losses_iter.append(lossD.item())
            G_losses_iter.append(lossG.item())

        
        G_losses_iter_mean = sum(G_losses_iter)/len(G_losses_iter)
        D_Losses_iter_mean = sum(D_Losses_iter)/len(D_Losses_iter)

        G_losses.append(G_losses_iter_mean)
        D_Losses.append(D_Losses_iter_mean)

        
        if epoch % 10 is 0:
            xgb_loss = accuracy_XGboost.CheckAccuracy(real_data_list, generated_data, feature_cols, label_col)
            xgb_losses = np.append(xgb_losses, xgb_loss)
            print(f'epoch: {epoch}, Accuracy: {xgb_loss}')
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                    % (epoch, num_epochs,
                        D_Losses_iter_mean, G_losses_iter_mean))
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

            accuracy_XGboost.PlotData(real_data_list, generated_data, feature_cols, label_col, seed=42, data_dim=2)
                # # Check how the generator is doing by saving G's output on fixed_noise
                # if (iters % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                #     with torch.no_grad():
                #         fake = netG(fixed_noise).detach().cpu()
                #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            


    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return xgb_losses, G_losses, D_Losses, generated_data, real_data_list

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



