import gin
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

@gin.configurable('BaseDecoder')
class BaseDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(BaseDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        for i, (in_size, out_size) in enumerate(zip(layer_dim_list[:-2], layer_dim_list[1:-2])):
            self.layers.add_module(name="Linear{:d}".format(i), 
                                module=nn.Linear(in_size, out_size))
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))

        self.layers.add_module(name="Linear{:d}".format(num_layers), 
                                module=nn.Linear(layer_dim_list[-2], layer_dim_list[-1]))
        # self.weight_init()

    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('ConvDecoder')
class ConvDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(ConvDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the ConvEncoder, every element except the first one 
            should be a quadruple, and the first element should be a tuple 
            contains (z_dim and the except dimension after view).
        '''
        num_layers = len(layer_dim_list) - 1
        self.layers = nn.Sequential()

        activation = getattr(nn, activation)

        self.z_dim, self.view_size = layer_dim_list[0]

        self.layers.add_module(name="Conv{:d}".format(0), 
                                module=nn.Conv2d(self.z_dim, self.view_size, 1))
                                
        for i, dim_tuple in enumerate(layer_dim_list[1:]):
            self.layers.add_module(name="Activation{:d}".format(i),
                                    module=activation(*act_param))
            self.layers.add_module(name="ConvTranspose2d{:d}".format(i+1), 
                                    module=nn.ConvTranspose2d(*dim_tuple))
            
        # self.weight_init()

    
    def forward(self, x):
        x = self.layers(x)
        return x

@gin.configurable('LSTMDecoder')
class LSTMDecoder(nn.Module):
    def __init__(self, layer_dim_list=gin.REQUIRED, activation='ReLU', act_param=[]):
        super(LSTMDecoder, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        ''' 
            In order to use the LSTMDecoder, mustset batch_first=True in LSTM layer, 
            and the shape of input should be (batch, timestep, features).
            activation/act_param in this model will not be used.
            layer_dim_list format: [input_size, hidden_size, num_layers, seq_len]
        '''
        self.input_size, self.hidden_size, self.num_layers,  self.seq_len = layer_dim_list
        self.layers = nn.Sequential()
        activation = getattr(nn, activation)

        self.layers.add_module(name="LSTM", 
                                module=nn.LSTM(
                                    input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers, 
                                    batch_first=True))
            
        # self.weight_init()
    
    def forward(self, x):
        x = x.repeat(1, self.seq_len)
        x = x.view(-1, self.seq_len, self.input_size)
        x, _ = self.layers(x)
        return x

@gin.configurable('ImageDecoder')
class ImageDecoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, input_size=gin.REQUIRED, n_channels=gin.REQUIRED, ngf=gin.REQUIRED, n_layers=gin.REQUIRED, norm=gin.REQUIRED):
    super(ImageDecoder, self).__init__()
    norm_layer = None
    if norm == 'BatchNorm2d':
        norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")


    ngf = ngf * (2 ** (n_layers - 2))
    layers = [nn.ConvTranspose2d(input_size, ngf, 4, 2, 1, bias=False)]
    if norm_layer:
        layers.append(norm_layer(ngf))
    layers.append(nn.ReLU(True))

    for i in range(1, n_layers - 1):
        layers.append(nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False))
        if norm_layer:
            layers.append(norm_layer(ngf // 2))
        layers.append(nn.ReLU(True))
        ngf = ngf // 2

    layers += [nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
                nn.Sigmoid()]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    if len(x.size()) == 2:
      x = x.view(*x.size(), 1, 1)
    x = self.main(x)
    return x

@gin.configurable('ImageDecoder_modified')
class ImageDecoderModified(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, input_size=gin.REQUIRED, n_channels=gin.REQUIRED, ngf=gin.REQUIRED, n_layers=gin.REQUIRED, norm=gin.REQUIRED):
    super(ImageDecoderModified, self).__init__()
    norm_layer = None
    if norm == 'BatchNorm2d':
        norm_layer = nn.BatchNorm2d
    elif norm == 'InstanceNorm2d':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'None':
        norm_layer = None
    else:
        raise NotImplementedError("Norm layer only support BN/IN/None")


    ngf = ngf * (2 ** (n_layers - 2))
    layers = [nn.ConvTranspose2d(input_size, ngf, 3, 1, bias=False)]
    if norm_layer:
        layers.append(norm_layer(ngf))
    layers.append(nn.ReLU(True))

    for i in range(1, n_layers - 1):
        layers.append(nn.ConvTranspose2d(ngf, ngf // 2, 3, 1, bias=False))
        if norm_layer:
            layers.append(norm_layer(ngf // 2))
        layers.append(nn.ReLU(True))
        ngf = ngf // 2

    layers += [nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False),
                nn.Sigmoid()]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    if len(x.size()) == 2:
      x = x.view(*x.size(), 1, 1)
    x = self.main(x)
    return x