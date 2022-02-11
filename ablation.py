import os
import logging
import logging.handlers
import copy
import gin


import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from main import *
from sfvae.utils import str2bool, mkdirs
from sfvae.exps import *
from sfvae.solvers import *
from sfvae.dataParser.downstreamDataset import *
from sfvae.models.vaes.others import *

# from sfvae.ablation.params_bikeNYC import *
# from sfvae.ablation.params_taxiNYC import *
from sfvae.ablation.params_taxiBJ import *
from sfvae.ablation import *

x_len_list = [1, 1, 1, 2, 2, 2, 3]
cpt_set_list = [[True, False, False], [False, True, False], [False, False, True], 
                [True, True, False], [True, False, True], [False, True, True], 
                [True, True, True]]

class CPTDataset(Dataset):

    def __init__(self, xs, y, x_len):
        assert (xs != [] and x_len > 0), "You should at least include one set of data in x."
        
        
        self.xs = []
        self.x_len = x_len
    
        for i in range(self.x_len):
            self.xs.append(torch.tensor(xs[i], dtype=torch.float))

        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.xs[0].size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        result = []
        
        for i in range(self.x_len):
            result.append(self.xs[i][idx])

        result.append(self.y[idx])
        
        return result

def load_solver(seed, model_name, gamma, log_dir, ckpt_dir, mode, model_index):
    temp_log_dir = log_dir + '{}_{}/'.format(seed, gamma)
    temp_ckpt_dir = ckpt_dir + '{}_{}/'.format(seed, gamma)
    mkdirs(temp_log_dir)
    mkdirs(temp_ckpt_dir)
    
    temp_module_name = "{}Solver".format(model_name)
    temp_func_name = temp_module_name[0].upper() + temp_module_name[1:]
    if temp_module_name in globals():
        model_solver = globals()[temp_module_name].__dict__[temp_func_name](mode, temp_log_dir, temp_ckpt_dir)
        model_solver.load_checkpoint(model_index)
    else:
        print("no such solver: {} solver found in this package".format(model_name))
        model_solver = None
    return model_solver

def exp_all_set(train_loader, test_loader, model, vae, optim_reg, early_stopping, loss_saving, x_len, max_epoch, s_info_list, t_info_list):
    model = train(train_loader, test_loader, model, vae, optim_reg, 
                        early_stopping, loss_saving, x_len, max_epoch, s_info_list, t_info_list)
    checkpoint = torch.load(early_stopping.path)
    model.load_state_dict(checkpoint)
    mae, rmse = test(test_loader, model, vae, optim_reg, x_len, s_info_list, t_info_list)
    return mae, rmse

def extarct_info_index(vae, train_loader, single_input):
    assert (single_input % 4) == 0, "single_input cannot be divided by 4"

    s_info_list = []
    t_info_list = []

    s_kld = torch.zeros((1, int(single_input / 4))).cuda()
    t_kld = torch.zeros((1, int(single_input / 4))).cuda()
    total_size = 0

    with torch.no_grad():
        for X, y in iter(train_loader):
            
            temp_size = X.size()[0]
            total_size += temp_size
            batch_size, n_frames_input, n_channels, H, W = X.size()


            _, (sm, ss), (tm, ts), _, _ = vae(X.cuda())
            s_kld += kl_divergence_per_z(sm, ss)
            t_kld += kl_divergence_per_z(tm, ts)
        
        
        s_kld = s_kld / total_size
        t_kld = t_kld / total_size

    for i in range(s_kld.size()[1]):
        if (s_kld[0, i] > 0.001):
            s_info_list.append(i)
    for i in range(t_kld.size()[1]):
        if (s_kld[0, i] > 0.001):
            t_info_list.append(i)       
    
    print("total number of informative features is: {}".format((len(s_info_list) + len(t_info_list))))
    return s_info_list, t_info_list, (len(s_info_list) + len(t_info_list)) * 2
    

def data_prep(dset_dir, dset_name):
    data = np.load(dset_dir + dset_name + '.npz')
    total_length = data['X'].shape[0]
    c_length = total_length - (7 * 48)

    trend = data['X'][:c_length]
    period = data['X'][(6 * 48):c_length + (6 * 48)]
    closeness = data['X'][(7 * 48):]
    y = data['y'][(7 * 48):]
    cpt_array = (closeness, period, trend)
    print(closeness.shape)
    print(period.shape)
    print(trend.shape)
    print(y.shape)
    return cpt_array, y, c_length

def single_set(run_number, seed, gamma, model_name, log_dir, ckpt_dir, mode, model_index, 
                x_len_list, cpt_array, cpt_set_list, valid_length, y_array, csv_path):
    
    print("start exps for run_number: {}, seed: {}, gamma: {}".format(run_number, seed, gamma))

    model_solver = load_solver(seed, model_name, gamma, log_dir, ckpt_dir, mode, model_index) 
    vae = model_solver.VAE
    
    loss_name_list = ['train_mae', 'train_rmse', 'valid_mae', 'valid_rmse']

    result_dict = {
        "run_number": 0,
        "seed":       0,
        "gamma":      0,
        "set_number": 0,   
        "feature_size": single_input,
        "full_feature_mae":  0,
        "full_feature_rmse":  0,
        "info_feature_mae":  0,
        "info_feature_rmse":  0,
    }

    for s_index, (x_len, cpt_setup) in enumerate(zip(x_len_list, cpt_set_list)):
        cpt_data = []
        for i, flag in enumerate(cpt_setup):
            if flag:
                cpt_data.append(cpt_array[i])

        train_dset = CPTDataset([temp[:valid_length] for temp in cpt_data], y_array[:valid_length], x_len)
        test_dset = CPTDataset([temp[valid_length:] for temp in cpt_data], y_array[valid_length:], x_len)

        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

        temp_dir = result_dir + '{}_{}_{}_c{}/'.format(run_number, seed, gamma, s_index)

        if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

        if s_index == 0:
            s_info_list, t_info_list, info_input = extarct_info_index(vae, train_loader, single_input)

        result_dict["run_number"] = run_number
        result_dict["seed"] = seed
        result_dict["gamma"] = gamma
        result_dict["set_number"] = s_index

        print("start full info exps with s_index: {}".format(s_index))
        model = MLP_Net([single_input * x_len] + model_args_full).cuda()
        optim_reg = optim.Adam(model.parameters(), lr=learning_rate_full)

        early_stopping = EarlyStopping(patience=50, path=temp_dir+'checkpoint_full.pt')
        loss_saving = LossSaving(loss_name_list, temp_dir+'training_loss_full.csv')
        mae, rmse = exp_all_set(train_loader, test_loader, model, vae, optim_reg, early_stopping, loss_saving, 
                                x_len, max_epoch, [], [])
        
        result_dict["full_feature_mae"] = mae
        result_dict["full_feature_rmse"] = rmse

        if info_input != 0:

            print("start info info exps with s_index: {}".format(s_index))
            model = MLP_Net([info_input * x_len] + model_args_info).cuda()
            optim_reg = optim.Adam(model.parameters(), lr=learning_rate_info)

            early_stopping = EarlyStopping(patience=50, path=temp_dir+'checkpoint_info.pt')
            loss_saving = LossSaving(loss_name_list, temp_dir+'training_loss_info.csv')
            mae, rmse = exp_all_set(train_loader, test_loader, model, vae, optim_reg, early_stopping, loss_saving, 
                                    x_len, max_epoch, s_info_list, t_info_list)

            result_dict["info_feature_mae"] = mae
            result_dict["info_feature_rmse"] = rmse
        else:
            print("skip info exps with s_index: {} since there isn't any informative features".format(s_index))
            result_dict["info_feature_mae"] = 0
            result_dict["info_feature_rmse"] = 0

        metric_to_csv(csv_path, result_dict)
        print("finished exps with s_index: {}".format(s_index))

def run_all_set(gin_file_path, dset_dir, dset_name, csv_folder_path, run_number_list, seed_list, gamma_list, base_path):
    gin.parse_config_file(gin_file_path)
    # gin.bind_parameter('StVAESolver.ckpt_load', True)
    gin.bind_parameter('StVAESolver.log_flag', False)

    cpt_array, y_array, c_length = data_prep(dset_dir, dset_name)
    train_length = int(c_length * train_portion)
    valid_length = int(c_length * (train_portion + valid_portion))

    mkdirs(csv_folder_path)
    csv_path = csv_folder_path + "full_info_exps.csv"

    print("finish data reading")
    
    for run_number in run_number_list:
        for seed in seed_list:
            for gamma in gamma_list:
                result_path = base_path + 'results/{}/'.format(run_number)
                ckpt_dir = result_path + 'models/'
                log_dir = result_path + 'log/'
                single_set(run_number, seed, gamma, model_name, log_dir, ckpt_dir, mode, model_index, 
                            x_len_list, cpt_array, cpt_set_list, valid_length, y_array, csv_path)


if __name__ == "__main__":
    run_all_set(gin_file_path, dset_dir, dset_name, csv_folder_path, run_number_list, seed_list, gamma_list, base_path)