import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseEncoder')
class BaseEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-1])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.linear_means = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        self.linear_log_var = nn.Linear(layer_dim_list[-2], layer_dim_list[-1])
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = self.linear_means(x)
        logvar = self.linear_log_var(x)
        return means, logvar


@gin.configurable('ConvEncoder')
class ConvEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(ConvEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the last one 
            should be a quadruple, and the last element should be a tuple 
            contains (z_dim and the except dimension after flatten).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        self.z_dim, self.view_size = layer_dim_list[-1]

        for i, dim_tuple in enumerate(layer_dim_list[:-1]):
            self.layers.add_module(name="Conv{:d}".format(i), 
                                    module=nn.Conv2d(*dim_tuple))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
        
        self.layers.add_module(name="Conv{:d}".format(num_layers), 
                                module=nn.Conv2d(self.view_size, 2*self.z_dim, 1))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        means = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return means, logvar

@gin.configurable('LSTMEncoder')
class LSTMEncoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(LSTMEncoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMEncoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [input_size, hidden_size, num_layers, latent_size]
        '''
        self.input_size, self.hidden_size, self.num_layers, self.latent_size  = layer_dim_list
        self.layers = nn.Sequential()
        activation = getattr(nn, activation)

        self.layers.add_module(name="LSTM", 
                                module=nn.LSTM(
                                    input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, 
                                    batch_first=True))
        
        self.linear_means = nn.Linear(self.hidden_size, self.latent_size)
        self.linear_log_var = nn.Linear(self.hidden_size, self.latent_size)
        # self.weight_init()

    def forward(self, x):
        [batch_size, seq_len, embed_size] = x.size()

        _, (_, final_state) = self.layers(x)

        final_state = final_state.view(self.num_layers, batch_size, self.hidden_size)
        final_state = final_state[-1]

        means = self.linear_means(final_state)
        logvar = self.linear_log_var(final_state)
        return means, logvar

@gin.configurable('ImageEncoder')
class ImageEncoder(nn.Module):
  '''
  Encodes images. Similar structure as DCGAN.
  '''
  def __init__(self, n_channels=gin.REQUIRED, output_size=gin.REQUIRED, ngf=gin.REQUIRED, n_layers=gin.REQUIRED, norm=gin.REQUIRED):
    super(ImageEncoder, self).__init__()

    norm_layer = None
    if norm == 'BatchNorm2d':
        norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")

    layers = [nn.Conv2d(n_channels, ngf, 4, 2, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)]

    for i in range(1, n_layers - 1):
        layers.append(nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False))
        if norm_layer:
            layers.append(norm_layer(ngf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        ngf *= 2

    layers += [nn.Conv2d(ngf, output_size, 4, 2, 1, bias=False)]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    x = self.main(x)
    return x

@gin.configurable('ImageEncoder_modified')
class ImageEncoderModified(nn.Module):
  '''
  Encodes images. Similar structure as DCGAN.
  '''
  def __init__(self, n_channels=gin.REQUIRED, output_size=gin.REQUIRED, ngf=gin.REQUIRED, n_layers=gin.REQUIRED, norm=gin.REQUIRED):
    super(ImageEncoderModified, self).__init__()

    norm_layer = None
    if norm == 'BatchNorm2d':
        norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")

    layers = [nn.Conv2d(n_channels, ngf, 4, 2, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)]

    for i in range(1, n_layers - 1):
        layers.append(nn.Conv2d(ngf, ngf * 2, 3, 1, bias=False))
        if norm_layer:
            layers.append(norm_layer(ngf * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        ngf *= 2

    layers += [nn.Conv2d(ngf, output_size, 3, 1, bias=False)]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    x = self.main(x)
    return x

@gin.configurable('ImageEncoder_ConvLSTM')
class ImageEncoderConvLSTM(nn.Module):
  '''
  Encodes images. Similar structure as DCGAN.
  '''
  def __init__(self, n_channels=gin.REQUIRED, output_size=gin.REQUIRED, ngf=gin.REQUIRED, n_layers=gin.REQUIRED, norm=gin.REQUIRED):
    super(ImageEncoderConvLSTM, self).__init__()

    self.norm_layer = None
    if norm == 'BatchNorm2d':
        self.norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        self.norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        self.norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")

    self.convlstm = ConvLSTM(input_dim=n_channels, 
                                hidden_dim=output_size, 
                                kernel_size=4,
                                num_layers=n_layers, 
                                batch_first=True)

  def forward(self, x):
    output_list, last_state_list = self.convlstm(x)
    print(len(output_list))
    print(len(last_state_list))
    print(output_list[0].size())
    print(last_state_list[0][0].size())
    if self.norm_layer:
        pass
    return x