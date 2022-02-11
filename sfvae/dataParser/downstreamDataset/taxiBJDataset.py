import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import *

class TaxiBJDataset(Dataset):
    """TaxiBJ dataset."""

    def __init__(self, X, y, x_norm=False, y_mode='eval'):
        self.X = X
        self.x_norm = x_norm
        self.x_scaler = None
    
        
        self.X = torch.tensor(self.X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        if y_mode == 'eval':
            self.y = self.y[:, :1024]

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        temp_x = self.X[idx]
        temp_y = self.y[idx]
        
        return temp_x, temp_y