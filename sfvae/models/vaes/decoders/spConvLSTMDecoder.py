import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from ...layers.convLSTM import ConvLSTM

@gin.configurable('SpConvLSTMDecoder')
class SpConvLSTMDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(SpConvLSTMDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvLSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, seq_len, convlstm_input_size]
        '''

        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.seq_len, self.latent_size, self.convlstm_input_size = layer_dim_list[0]
        self.kernel_size = tuple(self.kernel_size)

        activation = getattr(nn, activation)

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()
                                
        for i, dim_tuple in enumerate(layer_dim_list[1:]):
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
            self.layers.add_module(name="ConvTranspose2d{:d}".format(i+1), 
                                    module=nn.ConvTranspose2d(*dim_tuple))

        self.convlstm = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)
        self.before_conv_size = self.hidden_dim[0] * self.convlstm_input_size[0] * self.convlstm_input_size[1]
        self.trans = nn.Linear(self.latent_size, self.before_conv_size)
        
        # self.weight_init()
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.trans(x)
        x = x.repeat(1, self.seq_len)
        x = x.view(batch_size, self.seq_len, self.input_dim, self.convlstm_input_size[0], self.convlstm_input_size[1])
        x, _ = self.convlstm(x)

        x = x[0].view(batch_size * self.seq_len, self.hidden_dim[-1], self.convlstm_input_size[0], self.convlstm_input_size[1])
        x = self.layers(x)
        [_, embed_size, height, width] = x.size()
        x = x.view(batch_size, self.seq_len, embed_size, height, width)
        return x
