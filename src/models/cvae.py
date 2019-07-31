import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, data_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim*2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, data_dim),
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


def train_VAE(dataloader, data_dim:int,  h_dim: int, z_dim:int, lr:float, num_epochs:int, device='cpu'):
    
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    
    vae = VAE(data_dim, h_dim, z_dim)
    optimizerVAE = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    vae.to(device)
    # set the train mode
    #vae.train()

    # Loss function
    adversarial_loss = nn.BCELoss()

    print("Starting Training Loop...")
    start_time = time()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):

            optimizerVAE.zero_grad()

            real_samples = Variable(data.type(Tensor))

            loss.backward()
            optimizerVAE.step()
            
        
            if epoch % 50 is 0:
                print(f'Epoch {epoch}, Discriminator loss: {d_loss:.2f}, Generator loss: {g_loss:.2f}')

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