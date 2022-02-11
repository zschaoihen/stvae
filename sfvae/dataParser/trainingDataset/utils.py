import random

import numpy as np
import gin

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from . import betaVAEDataset, factorVAEDataset, stVAEDataset, VAEDataset

def get_solver_dataset(model_name):
    # if model_name == 'factorVAE':
    #     return trainingDataset.FactorVAEDataset
    # elif model_name == 'betaVAE':
    #     return trainingDataset.BetaVAEDataset
    temp_dset_name = "{}Dataset".format(model_name)
    temp_func_name = temp_dset_name[0].upper() + temp_dset_name[1:]
    if temp_dset_name in globals():
        return globals()[temp_dset_name].__dict__[temp_func_name]
    else:
        print("{} does not exist in this package".format(temp_dset_name))

@gin.configurable('get_loader', blacklist=['model_name', 'mode'])
def get_loader(model_name, mode, input_size=0, 
                dset_dir=gin.REQUIRED, dset_name=gin.REQUIRED, 
                batch_size=gin.REQUIRED, num_workers=gin.REQUIRED):
    dset = get_solver_dataset(model_name)
    
    if mode[0] == 'linear':
        with open(dset_dir + dset_name + '/train.npy', 'rb') as f:
            data = np.load(f)

        train_dset = dset(data)

    elif mode[0] == 'conv':
        image_size = input_size
        train_dset = return_data(dset_name, dset_dir, image_size)
    
    elif mode[0] == 'series':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'])

    elif mode[0] == 'spatio_temporal':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'])

    elif mode[0] == 'stgcn':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['adj_mx'])

    train_loader = DataLoader(train_dset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader