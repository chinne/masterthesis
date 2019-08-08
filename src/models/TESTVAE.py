import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
import numpy as np
from time import time


from ..common import accuracy_XGboost

class Encoder(nn.Module):
    def __init__(self, dataDim, compressDims, embeddingDim):
        super(Encoder, self).__init__()
        dim = dataDim
        seq = []
        for item in list(compressDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        self.seq = nn.Sequential(*seq)
        self.fc1 = nn.Linear(dim, embeddingDim)
        self.fc2 = nn.Linear(dim, embeddingDim)

    def forward(self, input):
        feature = self.seq(input)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar

class Decoder(nn.Module):
    def __init__(self, embeddingDim, decompressDims, dataDim):
        super(Decoder, self).__init__()
        dim = embeddingDim
        seq = []
        for item in list(decompressDims):
            seq += [
                nn.Linear(dim, item),
                nn.ReLU()
            ]
            dim = item
        
        seq.append(nn.Linear(dim, dataDim))
        seq.append(nn.Sigmoid())
        self.seq = nn.Sequential(*seq)
        
        self.sigma = nn.Parameter(torch.ones(dataDim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma



# class VAE(nn.Module):
#     '''
#     This class generates a variational autoencoder (VAE).
    
#     data_dim: The size of the data dimension as input and output for the VAE. 
#     encoder_hidden_dim: The size of the hidden dimension of the FNN of the encoder
#     decoder_hidden_dim: The size of the hidden dimension of the FNN of the decoder
#     z_dim:  
    
#     '''
#     def __init__(self, data_dim, encoder_hidden_dim, decoder_hidden_dim, z_dim):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(data_dim, encoder_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(encoder_hidden_dim, z_dim*2)
#         )
        
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, decoder_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(decoder_hidden_dim, data_dim),
#             nn.Sigmoid()
#         )
    
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         z = mu + std * eps
#         return z
    
#     def forward(self, x):
#         h = self.encoder(x)
#         mu, logvar = torch.chunk(h, 2, dim=1)
#         #mu, logvar = self.encoder(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1,30), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Calculate KLD
    return BCE + KLD

def trainVAE(dataloader, num_epochs:int, data_dim:int, feature_cols, label_col=[], embeddingDim=5, compressDims=(20,10), decompressDims=(20,10)):
    '''
    Training function for the VAE
    '''
    
    Tensor = torch.FloatTensor

    l2scale = 1e-5

    encoder = Encoder(data_dim, compressDims, embeddingDim)
    decoder = Decoder(embeddingDim, compressDims, data_dim)
    optimizerAE = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), weight_decay=l2scale)
    train_loss = []
    
    train_lost_list = []
    test_lost_list = []
    xgb_losses = []

    print("Starting Training Loop...")
    start_time = time()
    for epoch in range(num_epochs):
        train_loss = 0
        generated_data = []
        real_data_list = []
        for i, data in enumerate(dataloader):
            real = Variable(data.type(Tensor))
            # optimizer.zero_grad()            
            # recon_batch, mu, logvar = vae(data)
            # loss = loss_function(recon_batch, data, mu, logvar)
            # loss.backward()
            # train_loss += loss.item()
            # optimizer.step()    

            mu, std, logvar = encoder(real)
            eps = torch.randn_like(std)
            emb = eps * std + mu
            rec, sigmas = decoder(emb)
            loss = loss_function(rec, real, mu, logvar)
        
            loss.backward()
            train_loss += loss.item()
            optimizerAE.step()
            decoder.sigma.data.clamp_(0.01, 1.)
            # Save Losses for plotting later
            generated_data.extend(rec.detach().numpy())
            real_data_list.extend(real.numpy())

        if epoch % 10 == 0:
            xgb_loss = accuracy_XGboost.CheckAccuracy(real_data_list, generated_data, feature_cols, label_col)
            xgb_losses = np.append(xgb_losses, xgb_loss)
            print(f'epoch: {epoch}, Accuracy: {xgb_loss}')
            accuracy_XGboost.PlotData(real_data_list, generated_data, feature_cols, label_col, seed=42, data_dim=2)
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict()
            }, "models/vae/model_{}.tar".format(epoch))
       
    
        train_lost_list.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        







    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return train_lost_list


def generate(n:int, num_epochs:int, epoch:int, data_dim:int, embeddingDim=5, compressDims=(20,10), decompressDims=(20,10), batch_size=64):
    data_dim = data_dim
    decoder = Decoder(embeddingDim, compressDims, data_dim)

    ret = []
    data = []
    generated_data = []
    for i in range(num_epochs):
        checkpoint = torch.load("models/vae/model_{}.tar".format(epoch))
        decoder.load_state_dict(checkpoint['decoder'])
        decoder.eval()
        #decoder.to(self.device)

        steps = n // batch_size + 1
        
        # for i in range(steps):
        mean = torch.zeros(batch_size, embeddingDim)
        std = mean + 1
        noise = torch.normal(mean=mean, std=std)#.to(self.device)
        fake, sigmas = decoder(noise)
        fake = torch.tanh(fake)
        generated_data.extend(fake.detach().cpu().numpy())#.item())
        # data = np.concatenate(data, axis=0)
        # data = data[:n]
        #data = self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
        #ret.append(data)
    return generated_data