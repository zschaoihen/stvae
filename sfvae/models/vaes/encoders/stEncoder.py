import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .baseEncoder import ImageEncoder
from ...layers.baseLayers import TemporalConvLayer
from ...layers.convLSTM import ConvLSTM

@gin.configurable('StEncoder')
class StEncoder(nn.Module):
    '''
    The backbone model. CNN + 2D LSTM.
    Given an input st_raster, output the mean and standard deviation of the latent vectors.
    '''
    def __init__(self, image_out_size=gin.REQUIRED, image_out_channel=gin.REQUIRED,  
                hidden_size=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED):
        super(StEncoder, self).__init__()

        self.image_encoder = image_encoder()

        self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
        # Encoder
        self.encode_rnn = nn.LSTM(self.image_latent_size, hidden_size,
                                    num_layers=1, batch_first=True)

        # Beta
        self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

        self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


        self.image_out_size = image_out_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, input):
        batch_size, n_frames_input, n_channels, H, W = input.size()
        # encode each frame
        input_reprs = self.image_encoder(input.view(-1, n_channels, H, W))
        input_reprs = input_reprs.view(batch_size, n_frames_input, -1)

        assert input_reprs.size() == (batch_size, n_frames_input, self.image_latent_size), "wrong latent size!"
        output, hidden = self.encode_rnn(input_reprs)

        input_reprs = input_reprs.view(-1, self.image_latent_size)
        output = output.contiguous().view(-1, self.hidden_size)

        spatial_mu = self.spatial_mu_layer(input_reprs).view(batch_size, n_frames_input, self.output_size)
        spatial_sigma = self.spatial_sigma_layer(input_reprs).view(batch_size, n_frames_input, self.output_size)
        temporal_mu = self.temporal_mu_layer(output).view(batch_size, n_frames_input, self.output_size)
        temporal_sigma = self.temporal_sigma_layer(output).view(batch_size, n_frames_input, self.output_size)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma

@gin.configurable('RevisedStEncoder')
class RevisedStEncoder(nn.Module):
    '''
    The backbone model. Temporal Gated Convolutional + ResNet.
    Given an input st_raster, output the mean and standard deviation of the latent vectors.
    '''
    def __init__(self, temporal_module=gin.REQUIRED, conv3x3=gin.REQUIRED, image_out_size=gin.REQUIRED, 
                image_out_channel=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED):
        super(RevisedStEncoder, self).__init__()

        # Encoder
        self.image_encoder = image_encoder()

        self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
        self.image_out_channel = image_out_channel

        self.temporal_module = temporal_module()
        self.conv1 = conv3x3()
        self.conv2 = conv3x3()
        self.conv3 = conv3x3()

        # Beta
        self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

        self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


        self.image_out_size = image_out_size
        self.output_size = output_size


    def forward(self, x):
        batch_size, n_frames_input, n_channels, H, W = x.size()
        # encode each frame
        x_st = self.image_encoder(x.view(-1, n_channels, H, W))
        temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        # temporal gated conv
        x_t = self.temporal_module(temporal_input)
        x_t = x_t.contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        x = self.conv1(x_st)

        x += x_st

        x = self.conv2(x)
        x += x_t

        x_s = self.conv3(x)

        x_s = x_s.view(-1, self.image_latent_size)
        x_t = x_t.view(-1, self.image_latent_size)

        spatial_mu = self.spatial_mu_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        spatial_sigma = self.spatial_sigma_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        temporal_mu = self.temporal_mu_layer(x_t).view(batch_size, n_frames_input, self.output_size)
        temporal_sigma = self.temporal_sigma_layer(x_t).view(batch_size, n_frames_input, self.output_size)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma


@gin.configurable('EntangledStEncoder')
class EntangledStEncoder(nn.Module):
    '''
    The backbone model. Temporal Gated Convolutional + ResNet.
    Given an input st_raster, output the mean and standard deviation of the latent vectors.
    '''
    def __init__(self, temporal_module=gin.REQUIRED, conv3x3=gin.REQUIRED, image_out_size=gin.REQUIRED, 
                image_out_channel=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED):
        super(EntangledStEncoder, self).__init__()

        # Encoder
        self.image_encoder = image_encoder()

        self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
        self.image_out_channel = image_out_channel

        self.temporal_module = temporal_module()
        self.conv1 = conv3x3()
        self.conv2 = conv3x3()
        self.conv3 = conv3x3()

        # Beta
        self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

        self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


        self.image_out_size = image_out_size
        self.output_size = output_size

    def forward(self, x):
        batch_size, n_frames_input, n_channels, H, W = x.size()
        # encode each frame
        # x_st = self.image_encoder(x.view(-1, n_channels, H, W))
        # temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])
        x_st, pred = self.image_encoder(x)
        spatial_input = x_st.view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])
        
        # temporal gated conv
        # x_t = self.temporal_module(temporal_input)
        x_t = self.temporal_module(x_st)
        x_t = x_t.contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        x = self.conv1(spatial_input)

        x += spatial_input

        x = self.conv2(x)
        # x += x_t

        x_s = self.conv3(x)

        x_s = x_s.view(-1, self.image_latent_size)
        x_t = x_t.view(-1, self.image_latent_size)

        spatial_mu = self.spatial_mu_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        spatial_sigma = self.spatial_sigma_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        temporal_mu = self.temporal_mu_layer(x_t).view(batch_size, n_frames_input, self.output_size)
        temporal_sigma = self.temporal_sigma_layer(x_t).view(batch_size, n_frames_input, self.output_size)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma


# @gin.configurable('AttentionStEncoder')
# class AttentionStEncoder(nn.Module):
#     '''
#     The backbone model. Temporal Gated Convolutional + ResNet.
#     Given an input st_raster, output the mean and standard deviation of the latent vectors.
#     '''
#     def __init__(self, temporal_module=gin.REQUIRED, conv3x3=gin.REQUIRED, image_out_size=gin.REQUIRED, 
#                 image_out_channel=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED, num_heads=gin.REQUIRED):
#         super(AttentionStEncoder, self).__init__()

#         # Encoder
#         self.image_encoder = image_encoder()

#         self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
#         self.image_out_channel = image_out_channel

#         self.temporal_module = temporal_module()
#         self.conv1 = conv3x3()
#         self.conv2 = conv3x3()
#         self.conv3 = conv3x3()

#         # Beta
#         self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
#         self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

#         self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
#         self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


#         self.image_out_size = image_out_size
#         self.output_size = output_size

#         # attention
#         self.attention_1 = nn.MultiheadAttention(self.image_latent_size, num_heads)
#         self.attention_2 = nn.MultiheadAttention(self.image_latent_size, num_heads)
#         self.attention_3 = nn.MultiheadAttention(self.image_latent_size, num_heads)


#     def forward(self, x):
#         batch_size, n_frames_input, n_channels, H, W = x.size()
#         # encode each frame
#         x_st = self.image_encoder(x.view(-1, n_channels, H, W))
#         temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

#         # temporal gated conv
#         x_t = self.temporal_module(temporal_input)

#         x_t = x_t.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
#         x_t, _ = self.attention_1(x_t, x_t, x_t)
#         # x_t = x_t.contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])
#         x_t = x_t.transpose(0, 1).reshape(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])
#         x = self.conv1(x_st)

#         x += x_st

#         x = self.conv2(x)
#         x += x_t

#         x_s = self.conv3(x)

#         x_s = x_s.view(-1, self.image_latent_size)
#         x_t = x_t.view(-1, self.image_latent_size)

#         spatial_mu = self.spatial_mu_layer(x_s).view(batch_size, n_frames_input, self.output_size)
#         spatial_sigma = self.spatial_sigma_layer(x_s).view(batch_size, n_frames_input, self.output_size)
#         temporal_mu = self.temporal_mu_layer(x_t).view(batch_size, n_frames_input, self.output_size)
#         temporal_sigma = self.temporal_sigma_layer(x_t)
#         temporal_sigma = temporal_sigma.view(batch_size, n_frames_input, self.output_size)

#         return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma
    
#     def get_attn(self, x):
#         batch_size, n_frames_input, n_channels, H, W = x.size()
#         # encode each frame
#         x_st = self.image_encoder(x.view(-1, n_channels, H, W))
#         temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

#         # temporal gated conv
#         x_t = self.temporal_module(temporal_input)

#         x_t = x_t.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
#         x_t, x_t_w = self.attention_1(x_t, x_t, x_t)
#         return x_t, x_t_w


@gin.configurable('AttentionStEncoder')
class AttentionStEncoder(nn.Module):
    '''
    The backbone model. Temporal Gated Convolutional + ResNet.
    Given an input st_raster, output the mean and standard deviation of the latent vectors.
    '''
    def __init__(self, temporal_module=gin.REQUIRED, conv3x3=gin.REQUIRED, image_out_size=gin.REQUIRED, 
                image_out_channel=gin.REQUIRED, output_size=gin.REQUIRED, image_encoder=gin.REQUIRED, num_heads=gin.REQUIRED):
        super(AttentionStEncoder, self).__init__()

        # Encoder
        self.image_encoder = image_encoder()

        self.image_latent_size = image_out_size[0] * image_out_size[1] * image_out_channel
        self.image_out_channel = image_out_channel

        self.temporal_module = temporal_module()
        self.conv1 = conv3x3()
        self.conv2 = conv3x3()
        self.conv3 = conv3x3()

        # Beta
        self.spatial_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.spatial_sigma_layer = nn.Linear(self.image_latent_size, output_size)

        self.temporal_mu_layer = nn.Linear(self.image_latent_size, output_size)
        self.temporal_sigma_layer = nn.Linear(self.image_latent_size, output_size)


        self.image_out_size = image_out_size
        self.output_size = output_size

        # attention
        self.attention_1 = nn.MultiheadAttention(self.image_latent_size, num_heads)
        self.attention_2 = nn.MultiheadAttention(self.image_latent_size, num_heads)
        self.attention_3 = nn.MultiheadAttention(self.image_latent_size, num_heads)


    def forward(self, x):
        batch_size, n_frames_input, n_channels, H, W = x.size()
        # encode each frame
        x_st = self.image_encoder(x.view(-1, n_channels, H, W))
        temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        # temporal gated conv
        x_t = self.temporal_module(temporal_input)

        x_t = x_t.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
        x_t, _ = self.attention_1(x_t, x_t, x_t)
        x_t = x_t.transpose(0, 1).contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        x = self.conv1(x_st)

        x += x_st

        x = self.conv2(x)
        x += x_t

        x_s = self.conv3(x)

        x_s = x_s.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
        x_s, _ = self.attention_3(x_s, x_s, x_s)
        x_s = x_s.transpose(0, 1).contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        x_s = x_s.view(-1, self.image_latent_size)
        x_t = x_t.view(-1, self.image_latent_size)

        spatial_mu = self.spatial_mu_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        spatial_sigma = self.spatial_sigma_layer(x_s).view(batch_size, n_frames_input, self.output_size)
        temporal_mu = self.temporal_mu_layer(x_t).view(batch_size, n_frames_input, self.output_size)
        temporal_sigma = self.temporal_sigma_layer(x_t).view(batch_size, n_frames_input, self.output_size)

        return spatial_mu, spatial_sigma, temporal_mu, temporal_sigma
    
    def get_attn(self, x):
        batch_size, n_frames_input, n_channels, H, W = x.size()
        # encode each frame
        x_st = self.image_encoder(x.view(-1, n_channels, H, W))
        temporal_input = x_st.view(batch_size, n_frames_input, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        # temporal gated conv
        x_t = self.temporal_module(temporal_input)

        x_t = x_t.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
        x_t, x_t_w = self.attention_1(x_t, x_t, x_t)
        x_t = x_t.transpose(0, 1).contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])

        x = self.conv1(x_st)

        x += x_st

        x = self.conv2(x)
        x += x_t

        x_s = self.conv3(x)

        x_s = x_s.contiguous().view(batch_size, n_frames_input, -1).transpose(0, 1)
        x_s, x_s_w = self.attention_3(x_s, x_s, x_s)
        x_s = x_s.transpose(0, 1).contiguous().view(-1, self.image_out_channel, self.image_out_size[0], self.image_out_size[1])
        
        x_t = x_t.contiguous().view(batch_size, n_frames_input, -1)
        x_s = x_s.contiguous().view(batch_size, n_frames_input, -1)
        return (x_t, x_t_w), (x_s, x_s_w)


