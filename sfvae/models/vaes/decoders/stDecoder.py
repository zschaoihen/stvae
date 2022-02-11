import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .baseDecoder import ImageDecoder
from ...layers.baseLayers import AdaIN

@gin.configurable('StDecoder')
class StDecoder(nn.Module):
    def __init__(self, spatial_channel=gin.REQUIRED, image_in_size=gin.REQUIRED,
                temporal_channel=gin.REQUIRED, image_decoder=gin.REQUIRED):
        super(StDecoder, self).__init__()

        self.image_decoder = image_decoder()

        self.image_in_size = image_in_size
        self.spatial_channel = spatial_channel
        self.temporal_channel = temporal_channel
    
    def forward(self, spatial_z, temporal_z):

        batch_size, n_frames_input, _ = spatial_z.size()

        z = torch.cat((temporal_z, spatial_z), 2)
        z = z.view(batch_size * n_frames_input, (self.spatial_channel + self.temporal_channel), self.image_in_size[0], self.image_in_size[1])
        x = self.image_decoder(z)
        _, num_channel, H, W = x.size()
        x = x.view(batch_size, n_frames_input, num_channel, H, W)
        return x

@gin.configurable('RevisedStDecoder')
class RevisedStDecoder(nn.Module):
    def __init__(self, spatial_channel=gin.REQUIRED, image_in_size=gin.REQUIRED,
                temporal_channel=gin.REQUIRED, image_decoder=gin.REQUIRED):
        super(RevisedStDecoder, self).__init__()

        self.image_decoder = image_decoder()
        self.adain = AdaIN()

        self.image_in_size = image_in_size
        self.spatial_channel = spatial_channel
        self.temporal_channel = temporal_channel
    
    def forward(self, spatial_z, temporal_z):
        batch_size, n_frames_input, _ = spatial_z.size()

        spatial_z = spatial_z.view(batch_size * n_frames_input, self.spatial_channel, self.image_in_size[0], self.image_in_size[1])
        temporal_z = temporal_z.view(batch_size * n_frames_input, self.temporal_channel, self.image_in_size[0], self.image_in_size[1])

        z = self.adain(spatial_z, temporal_z)
        x = self.image_decoder(z)
        _, num_channel, H, W = x.size()
        x = x.view(batch_size, n_frames_input, num_channel, H, W)
        return x

