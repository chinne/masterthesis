import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


def train_VAE(dataloader, data_dim:int,  encoder_hidden_dim: int, decoder_hidden_dim: int, z_dim:int, lr:float, num_epochs:int, device='cpu'):
    '''
    Training function for the VAE
    '''
    
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    
    #Create VAE object
    vae = VAE(data_dim, encoder_hidden_dim, decoder_hidden_dim, z_dim) 
    
    #In case of an available supported gpu use it
    vae.to(device)
    
    #Construct optimizer for the VAE - in this case adam
    #lr: learning rate (default: 1e-3)
    #betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))
    
    G_losses = []
    D_losses = []
    
    # set the train mode
    #vae.train()

    # Creates a criterion that measures the Binary Cross Entropy between the target and the output
    adversarial_loss = nn.BCELoss()

    print("Starting Training Loop...")
    start_time = time()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            print('The variables i value is', i)
            print('The shape of the variable data is ', data.shape())

            #set gradients of all parameters of the optimizer to zero.
            optimizer.zero_grad()

            #get real data
            real_samples = Variable(data.type(Tensor))

            loss.backward()
            optimizerVAE.step()
            
        
            if epoch % 50 is 0:
                print(f'Epoch: {epoch} | Discriminator loss: {d_loss:.2f} | Generator loss: {g_loss:.2f}')

            # Save Losses for plotting later
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
        print(i+1, loss_1, loss_2)
        if i+1 in self.store_epoch:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict()
            }, "{}/model_{}.tar".format(self.working_dir, i+1))
    
    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
    return G_losses, D_losses





def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

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