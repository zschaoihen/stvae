import os
import numpy as np
import joblib
import pandas as pd
import csv

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def kl_divergence_per_z(mu, logvar):
    klds = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    total_kld = klds.sum(0)
    return total_kld

def metric_to_csv(path, loss_dict):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=loss_dict.keys()) 
            writer.writeheader() 
            writer.writerow(loss_dict)  
    else:
        with open(path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(loss_dict.values())

class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
class LossSaving():
    def __init__(self, loss_name_list, file_path):
        self.loss_dict = {}
        self.loss_name_list = loss_name_list
        self.file_path = file_path
        
        for loss_name in loss_name_list:
            self.loss_dict[loss_name] = []
    
    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self.loss_dict[str(key)]
    
    def __getitem__(self, key):
        return self.loss_dict[key]

    def __setitem__(self, key, val):
        self.loss_dict[key].append(val)
        
    def __call__(self, loss_list):
        assert len(self.loss_name_list) == len(loss_list), \
        "the length of the loss list should be {} and the order should be {}".format(len(self.loss_name_list), self.loss_name_list)
        
        for i in range(len(loss_list)):
            self.__setitem__(self.loss_name_list[i], loss_list[i])
            
    def save_loss(self):
        df_loss = pd.DataFrame.from_dict(self.loss_dict)
        df_loss.to_csv(self.file_path)

class MLP_Net(nn.Module):
    def __init__(self, layer_dim_list):
        super(MLP_Net, self).__init__()

        assert len(layer_dim_list) >= 2, "insufficient layer dims"
        num_layers = len(layer_dim_list) - 1
        layers = []

        for index in range(num_layers-1):
            layers.append(nn.Linear(layer_dim_list[index], layer_dim_list[index+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_dim_list[-2], layer_dim_list[-1]))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x