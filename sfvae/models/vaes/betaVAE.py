import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *

@gin.configurable('BetaVAE_network')
class BetaVAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED):
                    
        super(BetaVAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list

        self.encoder = encoder
        self.decoder = decoder

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
    Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017):
        encoder_dims = [
            (3, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 64, 4, 2, 1),
            (64, 64, 4, 2, 1),
            (64, 256, 4, 1),
            (10, 256),
        ]
        decoder_dims = [
            (10, 256), 
            (256, 64, 4),
            (64, 64, 4, 2, 1),
            (64, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 3, 4, 2, 1),
        ]
    Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018):
        encoder_dims = [
            (3, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (512, 256),
            (256, 256),
            (10, 256),
        ]
        decoder_dims = [
            (10, 256), 
            (256, 256),
            (256, 512),
            (256, 64, 4),
            (32, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 32, 4, 2, 1),
            (32, 3, 4, 2, 1),
        ]
'''