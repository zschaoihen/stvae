import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *
from .others import *

@gin.configurable('SpVAE_network')
class SpVAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED):
                    
        super(SpVAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list
        self.encoder_1 = encoder()
        self.encoder_2 = encoder()

        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def encode_(self, x):
        mu_1, logvar_1 = self.encoder_1(x)
        mu_2, logvar_2 = self.encoder_2(x)

        mu = torch.cat((mu_1, mu_2), 1)
        logvar = torch.cat((logvar_1, logvar_2), 1)
        return mu, logvar

    def decode_(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x, no_dec=False):
        mu, logvar = self.encode_(x)
        
        z = self.reparameterize(mu, logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode_(z)
            return x_recon, mu, logvar, z.squeeze()

@gin.configurable('RevisedSpVAE_network')
class RevisedSpVAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED, trans=gin.REQUIRED):
                    
        super(RevisedSpVAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list
        self.encoder_1 = encoder()
        self.encoder_2 = encoder()
        self.trans = trans()

        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def encode_(self, x):
        mu_1, logvar_1 = self.encoder_1(x)
        mu_2, logvar_2 = self.encoder_2(x)

        mu = torch.cat((mu_1, mu_2), 1)
        logvar = torch.cat((logvar_1, logvar_2), 1)
        return mu, logvar

    def decode_(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x, no_dec=False):
        mu, logvar = self.encode_(x)
        
        z = self.reparameterize(mu, logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode_(z)
            return x_recon, mu, logvar, z.squeeze()