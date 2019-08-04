import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from time import time

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
        self.seq = nn.Sequential(*seq)
        self.sigma = nn.Parameter(torch.ones(dataDim) * 0.1)

    def forward(self, input):
        return self.seq(input), self.sigma



class VAE(nn.Module):
    '''
    This class generates a variational autoencoder (VAE).
    
    data_dim: The size of the data dimension as input and output for the VAE. 
    encoder_hidden_dim: The size of the hidden dimension of the FNN of the encoder
    decoder_hidden_dim: The size of the hidden dimension of the FNN of the decoder
    z_dim:  
    
    '''
    def __init__(self, data_dim, encoder_hidden_dim, decoder_hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, data_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        #mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1,30), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Calculate KLD
    return BCE + KLD

def trainVAE(dataloader, num_epochs:int, data_dim:int, feature_cols, label_col=[], encoder_hidden_dim=20, decoder_hidden_dim=20, z_dim=5):
    '''
    Training function for the VAE
    '''
    
    Tensor = torch.FloatTensor
    vae = VAE(data_dim, encoder_hidden_dim, decoder_hidden_dim, z_dim)
    # set the train mode
    vae.train()
    train_loss = []
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    train_lost_list = []
    test_lost_list = []
    

    print("Starting Training Loop...")
    start_time = time()
    for epoch in range(num_epochs):
        train_loss = 0
        for i, data in enumerate(dataloader):
            data = Variable(data.type(Tensor))
            optimizer.zero_grad()            
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()           
        
        train_lost_list.append(train_loss)
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        







    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return train_lost_list



def test_vae():
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            data = Variable(data.type(Tensor))
            recon_batch, mu, logvar = vae(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = 



def generate(self, n):
        data_dim = self.transformer.output_dim
        decoder = Decoder(self.embeddingDim, self.compressDims, data_dim).to(self.device)

        ret = []
        for epoch in self.store_epoch:
            checkpoint = torch.load("{}/model_{}.tar".format(self.working_dir, epoch))
            decoder.load_state_dict(checkpoint['decoder'])
            decoder.eval()
            decoder.to(self.device)

            steps = n // self.batch_size + 1
            data = []
            for i in range(steps):
                mean = torch.zeros(self.batch_size, self.embeddingDim)
                std = mean + 1
                noise = torch.normal(mean=mean, std=std).to(self.device)
                fake, sigmas = decoder(noise)
                fake = torch.tanh(fake)
                data.append(fake.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            data = data[:n]
            data = self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())
            ret.append((epoch, data))
        return ret


