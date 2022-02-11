import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *

@gin.configurable('VanillaVAE_network')
class VAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED, mode='conv'):
                    
        super(VAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list

        self.mode = mode

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

    def forward(self, x):
        mu, logvar = self.encode_(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_(z), mu, logvar