import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from ...layers.convLSTM import ConvLSTM

@gin.configurable('ConvLSTMEncoder')
class ConvLSTMEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(ConvLSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, image_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.image_size = layer_dim_list
        self.kernel_size = tuple(self.kernel_size)

        activation = getattr(nn, activation)

        # self.layers = nn.Sequential()
        # self.layers.add_module(name="ConvLSTM", 
        #                         module=ConvLSTM(
        #                             input_dim=self.input_dim, 
        #                             hidden_dim=self.hidden_dim, 
        #                             kernel_size=self.kernel_size,
        #                             num_layers=self.num_layers, 
        #                             batch_first=True))
        self.layers = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)
        
        self.latent_size = self.hidden_dim[-1] * self.image_size[0] * self.image_size[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        [batch_size, seq_len, embed_size, height, width] = x.size()
        _, pred = self.layers(x)
        final_state = pred[0][0].view(batch_size, self.latent_size)

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar

@gin.configurable('RevisedConvLSTMEncoder')
class RevisedConvLSTMEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(RevisedConvLSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [input_dim, hidden_dim, kernel_size, num_layers, convlstm_input_size]
        '''
        self.input_dim, self.hidden_dim, self.kernel_size, self.num_layers, self.convlstm_input_size = layer_dim_list[-1]
        self.kernel_size = tuple(self.kernel_size)

        activation = getattr(nn, activation)

        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        for i, dim_tuple in enumerate(layer_dim_list[:-1]):
            self.layers.add_module(name="Conv{:d}".format(i), 
                                    module=nn.Conv2d(*dim_tuple))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))

        self.convlstm = ConvLSTM(input_dim=self.input_dim, 
                                hidden_dim=self.hidden_dim, 
                                kernel_size=self.kernel_size,
                                num_layers=self.num_layers, 
                                batch_first=True)
        
        self.latent_size = self.hidden_dim[-1] * self.convlstm_input_size[0] * self.convlstm_input_size[1]
        self.linear_means = nn.Linear(self.latent_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.latent_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        [batch_size, seq_len, embed_size, height, width] = x.size()
        x = x.view(batch_size * seq_len, embed_size, height, width)
        x = self.layers(x)
        x = x.view(batch_size, seq_len, self.hidden_dim[-1], self.convlstm_input_size[0], self.convlstm_input_size[1])
        _, pred = self.convlstm(x)
        final_state = pred[0][0].view(batch_size, self.latent_size)

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar