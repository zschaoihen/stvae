import gin

config_name = "run_bikeNYC"
scaler_path = "D:/Separate_factors/data/processed/bikeNYC/BikeNYC_scaler.pkl"

# double size
gin_file_path = './configs/{}/main_setup_1.gin'.format(config_name)
single_input = 288
run_number_list = [301, 304]

# # single size
# gin_file_path = './configs/{}/main_setup.gin'.format(config_name)
# single_input = 144
# run_number_list = [320, 321, 322, 323]

#curent
base_path = './'
model_name = "stVAE"

model_index = -1
gamma_list = [0.4, 1.6]
seed_list = [35, 42]
mode = ['spatio_temporal', 'revised']

MAX_VALUE = 299
train_portion = 0.7
valid_portion = 0.1

result_dir = 'D:/Separate_factors/repo/results/ablation_checkpoint/bikeNYC/'

dset_dir = 'D:/Separate_factors/data/processed/bikeNYC/'
dset_name = 'BikeNYC_st'

random_seed = 42
max_epoch = 500
batch_size = 128

# # full features model args
# model_args_full = [500, 1000, 2000, 1000, 200]
# learning_rate_full = 0.0005

# # info features model args
# model_args_info = [500, 1000, 2000, 1000, 200]
# learning_rate_info = 0.0001

# # full features model args
# model_args_full = [500, 1000, 1000, 200]
# learning_rate_full = 0.0003

# # info features model args
# model_args_info = [500, 1000, 1000, 200]
# learning_rate_info = 0.0001

# # full features model args
# model_args_full = [500, 1200, 200]
# learning_rate_full = 0.0003

# # info features model args
# model_args_info = [500, 1200, 200]
# learning_rate_info = 0.0001

# full features model args
model_args_full = [200, 500, 1200, 500, 200]
learning_rate_full = 0.0005

# info features model args
model_args_info = [200, 500, 1200, 500, 200]
learning_rate_info = 0.0003

csv_folder_path = "D:/Separate_factors/repo/results/full_info/bikeNYC/"

# double size
