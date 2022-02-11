#################################################################
CITY = 'bikeNYC'
trainRatio = 0.8  # train/test
SPLIT = 0.2  # train/val
MAX_VALUE = 307.0
freq = '30min'
INTERVAL = 30
DAYTIMESTEP = int(24 * 60 / INTERVAL)
HEIGHT = 10
WIDTH = 20
local_image_size = 9
grid_size = 1  # for graph
feature_len = DAYTIMESTEP + 7 + 2
toponet_len = 32
#################################################################

TIMESTEP = 6

BATCHSIZE = 1024  # all:(T-TIMESTEP)*32*32, should be a divisor
LOSS = 'mse'
OPTIMIZER = 'adam'
EPOCH = 200
LR = 0.0001

raw_dataPath = '../../../data/raw/{}/'.format(CITY)  # used by preprocess
processed_dataPath = '../../../data/processed/{}/'.format(CITY)  # used by preprocess
flow_path_lst = [raw_dataPath + 'BikeNYC_raw.npy']
timeFile = raw_dataPath + 'BikeNYC_timestamps.npy'



processed_flow_path = processed_dataPath + 'BikeNYC_st.npz'
scaler_path = processed_dataPath + 'BikeNYC_scaler.pkl'