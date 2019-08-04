import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from time import time

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(29, 20)
        self.fc2m = nn.Linear(20, 10) # use for mean
        self.fc2s = nn.Linear(20, 10) # use for standard deviation
        
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, 29)
        
    def reparameterize(self, log_var, mu):
        s = torch.exp(0.5*log_var)
        eps = torch.rand_like(s) # generate a iid standard normal same shape as s
        return eps.mul(s).add_(mu)
        
    def forward(self, input):
        x = input
        x = torch.relu(self.fc1(x))
        log_s = self.fc2s(x)
        m = self.fc2m(x)
        z = self.reparameterize(log_s, m)
        
        x = self.decode(z)
        
        return x, m, log_s
    
    def decode(self, z):
        x = torch.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))
        return x

def loss_function(real_data, recon_x, mu, log_var):
    print(recon_x.size(), real_data.size())
    CE = F.binary_cross_entropy(recon_x, torch.argmax(real_data, dim=-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KLD + CE

def train(dataloader, num_epochs:int, data_dim:int, feature_cols, label_col=[], embeddingDim=128, compressDims=(128, 128), decompressDims=(128, 128)):
    '''
    Training function for the VAE
    '''
    
    Tensor = torch.FloatTensor
    
    vae = VAE()
    train_loss = []
    optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.5, 0.999))
    print("Starting Training Loop...")
    start_time = time()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):

            #set gradients of all parameters of the optimizer to zero.
            

            #get real data
            real_data = Variable(data.type(Tensor))
            optimizer.zero_grad()
            recon_image, s, mu = vae(real_data)


            loss = loss_function(real_data, recon_image, mu, s)
            loss.backward()
            train_loss.append(loss.item() / len(real_data))
            optimizer.step()           
    plt.plot(train_loss)
    plt.show()
#              #decoder.sigma.data.clamp_(0.01, 1.)
#    print('====> Epoch: {} Average loss: {:.4f}'.format(
#          epoch, train_loss / len(dataloader.dataset)))
    

    




    end_time = time()
    seconds_elapsed = end_time - start_time
    print('It took ', seconds_elapsed)
#    return losses








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