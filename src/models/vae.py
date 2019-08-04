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
        std = logvar.mul(0.5).exp_()
        esp = to_var(torch.randn(*mu.size()))
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def train(dataloader, num_epochs:int, data_dim:int, feature_cols, label_col=[], embeddingDim=128, compressDims=(128, 128), decompressDims=(128, 128)):
    '''
    Training function for the VAE
    '''
    
    Tensor = torch.FloatTensor
    
    encoder = Encoder(data_dim, compressDims, embeddingDim)
    decoder = Decoder(embeddingDim, compressDims, data_dim)

    def loss_function(recon_x, x, sigmas, mu, logvar):
        print(recon_x.size(), x.size())
        recon_loss = F.cross_entropy(recon_x, x.view(-1,29), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Calculate KLD
        return BCE + KLD


    loss_factor = 2
    l2scale = 1e-5
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), weight_decay=l2scale)
    #Create VAE object
    #vae = VAE(data_dim, encoder_hidden_dim, decoder_hidden_dim, z_dim) 
    
    #In case of an available supported gpu use it
    #vae.to(device)
    
    #Construct optimizer for the VAE - in this case adam
    #lr: learning rate (default: 1e-3)
    #betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    #optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
    
    losses = []
    D_losses = []
    
    # set the train mode
    #vae.train()

    # Creates a criterion that measures the Binary Cross Entropy between the target and the output
    #adversarial_loss = nn.BCELoss()

    print("Starting Training Loop...")
    start_time = time()
    train_loss = 0
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):

            #set gradients of all parameters of the optimizer to zero.
            optimizer.zero_grad()

            #get real data
            real = Variable(data.type(Tensor))
            mu, std, logvar = encoder(real)

            eps = torch.randn_like(std)
            emb = eps * std + mu #sample z
            rec, sigmas = decoder(emb) 

            Loss = loss_function(rec, real, sigmas, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()           

            #decoder.sigma.data.clamp_(0.01, 1.)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
        







    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return losses





def loss_function(recon_x, x, mu, logvar):
    
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 64), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


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