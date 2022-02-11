import math
import os
import datetime as dt

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from Param_bikeNYC import *

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def build_data(flow_data, date_list, sequence_len=6):
    assert flow_data.shape[0] == len(date_list), "length of the flow data is not consitent with the date data."
    flow_data = flow_data.transpose(0, 3, 1, 2)

    data_size, num_channel, H, W = flow_data.shape
    length = sequence_len + 1

    X_size = ((data_size-length+1), length, num_channel, H, W)
    date_size = ((data_size-length+1), 4)

    data = np.zeros(X_size)
    dates = np.zeros(date_size)

    for index in range(X_size[0]):
        date_str = date_list[index]
        year, month, date, idx = date_str[:5], date_str[5:7], date_str[7:9], date_str[-2:]
        dates[index] = np.array([year, month, date, idx]).reshape(1, 4)
        
        data[index] = flow_data[index:index+length]
    
    print("X + y shape = {}".format(data.shape))
    print("date shape = {}".format(dates.shape))
        
    return data[:, :sequence_len], data[:, -1], dates

##############################################################
if __name__ == '__main__':
    # build 9*9 image
    mkdir(processed_dataPath)
    print('load data from: {}'.format(raw_dataPath))

    temp_X = []
    temp_y = []
    temp_date = []

    date_data = np.load(timeFile, allow_pickle=True)
    # for idx, file_path in enumerate(flow_path_lst):   
    #     print("start processing file: {}".format(file_path))
    #     flow_data = np.load(file_path)
    #     date_list = date_data[idx]

    #     X, y, date = build_data(flow_data, date_list, sequence_len=TIMESTEP)
    #     temp_X.append(X)
    #     temp_y.append(y)
    #     temp_date.append(date)
    file_path = flow_path_lst[0]
    print("start processing file: {}".format(file_path))
    flow_data = np.load(file_path)
    date_list = date_data

    X, y, date = build_data(flow_data, date_list, sequence_len=TIMESTEP)
    temp_X.append(X)
    temp_y.append(y)
    temp_date.append(date)


    X = np.vstack(temp_X)
    y = np.vstack(temp_y)
    date = np.vstack(temp_date)

    scaler = MinMaxScaler() 
    shape = X.shape
    X = X.reshape(-1, 1)
    X = scaler.fit_transform(X)
    print(X.max())
    print(X.min())
    X = X.reshape(shape)
    
    shape = y.shape
    y = y.reshape(-1, 1)
    y = scaler.transform(y)
    y = y.reshape(shape)
    print(y.max())
    print(y.min())

    print("total number of X are {}".format(X.shape[0]))

    np.savez(processed_flow_path, X=X, y=y, date=date)
    joblib.dump(scaler, scaler_path) 