import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from ...layers.STGCN import *

@gin.configurable('STGCNEncoder')
class STGCNEncoder(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(STGCNEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [num_nodes, num_features, num_input_time_steps]
        '''
        self.num_nodes, self.num_features, self.num_timesteps_input = layer_dim_list[0]
        # self.kernel_size = tuple(self.kernel_size)
        activation = getattr(nn, activation)

        num_layers = len(layer_dim_list) - 1
        # self.layers = nn.Sequential()

        block_l1 = layer_dim_list[1]
        block_l2 = layer_dim_list[2]
        block_l3 = layer_dim_list[3]
        self.block1 = STGCNBlock(in_channels=self.num_features, out_channels=block_l1[1],
                                 spatial_channels=block_l1[2], num_nodes=self.num_nodes)
        self.block2 = STGCNBlock(in_channels=block_l2[0], out_channels=block_l2[1],
                                 spatial_channels=block_l2[2], num_nodes=self.num_nodes)
        self.last_temporal = TimeBlock(in_channels=block_l3[0], out_channels=block_l3[1], 
                                 kernel_size=block_l3[2])
#         self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
#         self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
#                                num_timesteps_output)
        # self.fully = nn.Linear((num_timesteps_input) * 64,
        #                        num_timesteps_output)
        
        self.latent_size = self.num_timesteps_input * self.num_nodes * block_l3[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        A_hat, x = x
        [batch_size, num_nodes, num_timesteps_input, num_features] = x.size()
        out1 = self.block1(x, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = out3.reshape((out3.shape[0], -1))

        means = self.linear_means(out4)
        logvar = self.linear_log_var(out4)
        return means, logvar
