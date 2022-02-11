import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .encoders import *
from .decoders import *
from .others import *

@gin.configurable('StVAE_network')
class StVAE(nn.Module):
    def __init__(self, encoder=gin.REQUIRED, decoder=gin.REQUIRED):
                    
        super(StVAE, self).__init__()

        # assert type(encoder_layer_dims) == list
        # assert type(decoder_layer_dims) == list
        self.encoder = encoder()
        self.decoder = decoder()

    def reparameterize(self, mu, logvar):
        batch_size, n_frames_input, latent_size = mu.size()
        mu = mu.view(batch_size * n_frames_input, -1)
        logvar = logvar.view(batch_size * n_frames_input, -1)

        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())

        z = mu + std*eps
        return z.view(batch_size, n_frames_input, latent_size)

    def encode_(self, x):
        spatial_mu, spatial_sigma, temporal_mu, temporal_sigma = self.encoder(x)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma

    def decode_(self, spatial_z, temporal_z):
        x = self.decoder(spatial_z, temporal_z)
        return x

    def forward(self, x, no_dec=False):
        spatial_mu, spatial_sigma, temporal_mu, temporal_sigma = self.encode_(x)
        
        spatial_z = self.reparameterize(spatial_mu, spatial_sigma)
        temporal_z = self.reparameterize(temporal_mu, temporal_sigma)

        batch_size, n_frames_input, _ = spatial_z.size()
        
        if no_dec:
            spatial_z = torch.flatten(spatial_z, start_dim=1)
            temporal_z = torch.flatten(temporal_z, start_dim=1)
            z = torch.cat((spatial_z, temporal_z), 1)
            return z
        else:
            x_recon = self.decode_(spatial_z, temporal_z)
            flat_spatial_z = torch.flatten(spatial_z, start_dim=1)
            flat_temporal_z = torch.flatten(temporal_z, start_dim=1)

            spatial_mu = torch.flatten(spatial_mu, start_dim=1)
            spatial_sigma = torch.flatten(spatial_sigma, start_dim=1)
            temporal_mu = torch.flatten(temporal_mu, start_dim=1)
            temporal_sigma = torch.flatten(temporal_sigma, start_dim=1)

            z = torch.cat((flat_spatial_z, flat_temporal_z), 1)
            return x_recon, (spatial_mu, spatial_sigma), (temporal_mu, temporal_sigma), (spatial_z, temporal_z), z