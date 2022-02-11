import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *
from .others import *

@gin.configurable('FactorVAE_network')
class FactorVAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED):
                    
        super(FactorVAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list

        self.encoder = encoder()
        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std*eps

    def encode_(self, x):
        mu, logvar = self.encoder(x)
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

'''
    The discriminator setup for FactorVAE is:
        activation=nn.LeakyReLU, act_param=[0.2, True]
    Model proposed in original factor-VAE paper:
        encoder_dims = [
            (1, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 64, 4, 2, 1),
            (64, 64, 4, 2, 1),
            (64, 128, 4, 1),
            (10, 128),
        ]
        decoder_dims = [
            (10, 128), 
            (128, 64, 4),
            (64, 64, 4, 2, 1),
            (64, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 3, 4, 2, 1),
        ]
'''