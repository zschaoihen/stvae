import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# informative train-test
def train_single(train_loader, optim_reg, model, vae, x_len, s_info_list=[], t_info_list=[]):
    model.train()
    
    total_size = 0
    total_rmse = 0
    total_rmse = 0
    total_mae = 0
    length = 0
    
    for batch_data in iter(train_loader):
        x_in_list = []
        optim_reg.zero_grad()

        y = batch_data[-1]
        y = y.cuda()[:, 0, :]
        y = torch.flatten(y, start_dim=1)
        
        temp_size = batch_data[0].size()[0]
        total_size += temp_size
        length = y.shape[-1]
        batch_size, n_frames_input, n_channels, H, W = batch_data[0].size()
        
        for i in range(x_len):
            _, (sm, ss), (tm, ts), _, _ = vae(batch_data[i].cuda())
            
            if s_info_list != [] or t_info_list != []:
                x_in_list += [sm[:, s_info_list], ss[:, s_info_list], tm[:, t_info_list], ts[:, t_info_list]]
            else:
                x_in_list += [sm, ss, tm, ts]
        x_in = torch.cat(x_in_list, 1)

        pred = model(x_in)
        pred = torch.flatten(pred, start_dim=1)

#         mse_loss = F.mse_loss(y, pred)
#         mse_loss.backward(retain_graph=True)
        mae_loss = F.l1_loss(y, pred)
        mae_loss.backward(retain_graph=True)
        optim_reg.step()
        
        total_rmse += F.mse_loss(y, pred, reduction='mean').item() * temp_size
        total_mae += F.l1_loss(y, pred, reduction='mean').item() * temp_size
    
    rmse = ((total_rmse / total_size) ** 0.5) 
    mae = (total_mae / total_size)    

    #     print("epoch: {}, MAE loss: {}".format(current_epoch, mae_loss * max_value))
    #     print("epoch: {}, RMSE loss: {}".format(current_epoch, rmse_loss * max_value))   
    return mae, rmse
        

def valid_single(valid_loader, model, vae, x_len, s_info_list=[], t_info_list=[]):
    model.eval()
    
    total_size = 0
    total_rmse = 0
    total_rmse = 0
    total_mae = 0
    length = 0
    
    for batch_data in iter(valid_loader): 
        x_in_list = []
        
        y = batch_data[-1]
        y = y.cuda()[:, 0, :]
        y = torch.flatten(y, start_dim=1)
        
        temp_size = batch_data[0].size()[0]
        total_size += temp_size
        length = y.shape[-1]
        batch_size, n_frames_input, n_channels, H, W = batch_data[0].size()
        
        for i in range(x_len):
            _, (sm, ss), (tm, ts), _, _ = vae(batch_data[i].cuda())
            if s_info_list != [] or t_info_list != []:
                x_in_list += [sm[:, s_info_list], ss[:, s_info_list], tm[:, t_info_list], ts[:, t_info_list]]
            else:
                x_in_list += [sm, ss, tm, ts]

        x_in = torch.cat(x_in_list, 1)
        
        pred = model(x_in)

        total_rmse += F.mse_loss(y, pred, reduction='mean').item() * temp_size
        total_mae += F.l1_loss(y, pred, reduction='mean').item() * temp_size
        
#     print(total_size)

    rmse = ((total_rmse / total_size) ** 0.5)
    mae = (total_mae / total_size)
#     print("MAE loss: {}".format(mae * max_value))
#     print("RMSE loss: {}".format(rmse * max_value))
    return mae, rmse


def train(train_loader, valid_loader, model, vae, optim_reg, early_stopping, loss_saving, x_len, max_epoch, s_info_list=[], t_info_list=[]):
    model.train()
    vae.eval()
    
    for current_epoch in range(max_epoch):
        train_mae, train_rmse = train_single(train_loader, optim_reg, model, vae, x_len, s_info_list, t_info_list)
        valid_mae, valid_rmse = valid_single(valid_loader, model, vae, x_len, s_info_list, t_info_list)
        
        # print("epoch: {}, Train MAE loss: {}, Train  RMSE loss: {}, Valid MAE loss: {}, Valid  RMSE loss: {}"\
        #      .format(current_epoch, round(train_mae, 3), round(train_rmse, 3), round(valid_mae, 3), round(valid_rmse, 3)))
        
#         early_stopping(valid_rmse, model)
        early_stopping(valid_mae, model)
        loss_saving([train_mae, train_rmse, valid_mae, valid_rmse])
        if early_stopping.early_stop:
            loss_saving.save_loss()
            break
    
    return model
    
    
def test(test_loader, model, vae, optim_reg, x_len, s_info_list=[], t_info_list=[]):
    model.eval()
    vae.eval()
    
    total_size = 0
    total_rmse = 0
    total_rmse = 0
    total_mae = 0
    length = 0
    
    x_in_list = []
    
    for batch_data in iter(test_loader):
#         C, P, T, y = next(iter(train_loader))
            
        y = batch_data[-1]
        y = y.cuda()[:, 0, :]
        y = torch.flatten(y, start_dim=1)
        
        temp_size = batch_data[0].size()[0]
        total_size += temp_size
        length = y.shape[-1]
        batch_size, n_frames_input, n_channels, H, W = batch_data[0].size()
        
#         print(X.size())
        for i in range(x_len):
            _, (sm, ss), (tm, ts), _, _ = vae(batch_data[i].cuda())
            if s_info_list != [] or t_info_list != []:
                x_in_list += [sm[:, s_info_list], ss[:, s_info_list], tm[:, t_info_list], ts[:, t_info_list]]
            else:
                x_in_list += [sm, ss, tm, ts]
        
        x_in = torch.cat(x_in_list, 1)
        
        pred = model(x_in)

        total_rmse += F.mse_loss(y, pred, reduction='mean').item() * temp_size
#         print(total_rmse)
        total_mae += F.l1_loss(y, pred, reduction='mean').item() * temp_size
    
        x_in_list = []
        
    # print(total_size)

    rmse = (total_rmse / total_size) ** 0.5
    mae = (total_mae / total_size)
    return mae, rmse