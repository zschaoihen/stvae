import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from ...layers.STGCN import STGCNBlock, TimeBlock

@gin.configurable('STGCNDecoder')
class STGCNDecoder(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(STGCNDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvLSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [num_nodes, num_features, num_timesteps_output]
        '''
        self.num_nodes, self.num_features, self.num_timesteps_output = layer_dim_list[0]
        # self.kernel_size = tuple(self.kernel_size)
        activation = getattr(nn, activation)

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        block_l1 = layer_dim_list[1]
        block_l2 = layer_dim_list[2]
        block_l3 = layer_dim_list[3]

        self.last_temporal = TimeBlock(in_channels=block_l1[0], out_channels=block_l1[1], 
                                        kernel_size=block_l1[2])
        self.block1 = STGCNBlock(in_channels=block_l2[0], out_channels=block_l2[1],
                                 spatial_channels=block_l2[2], num_nodes=self.num_nodes)
        self.block2 = STGCNBlock(in_channels=block_l3[0], out_channels=block_l3[1],
                                 spatial_channels=block_l3[2], num_nodes=self.num_nodes)

        
        self.latent_size = self.num_timesteps_output * self.num_nodes * block_l1[0]
        
        # self.weight_init()
    
    def forward(self, x):
        A_hat, x = x
        batch_size = x.size()[0]
        # x = x.repeat(1, self.num_timesteps_output)
        x = x.view(batch_size, self.num_nodes, self.num_timesteps_output, self.num_features)   
        x = self.last_temporal(x)
        out1 = self.block1(x, A_hat)
        out2 = self.block2(out1, A_hat)
        
        return out2