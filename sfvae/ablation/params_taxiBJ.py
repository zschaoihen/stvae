config_name = "run_107"
scaler_path = "D:/Separate_factors/data/processed/taxiBJ/TaxiBJ_scaler.pkl"

# double size
gin_file_path = './configs/{}/main_setup_1.gin'.format(config_name)
single_input = 384
run_number_list = [205, 206, 207, 208]

# # # single size
# gin_file_path = './configs/{}/main_setup.gin'.format(config_name)
# single_input = 192
# run_number_list = [220, 221, 222, 223]

#curent
base_path = './'
model_name = "stVAE"

model_index = -1
gamma_list = [0.4, 1.6]
seed_list = [35, 42]
mode = ['spatio_temporal', 'revised']


MAX_VALUE = 1292
train_portion = 0.7
valid_portion = 0.1

result_dir = 'D:/Separate_factors/repo/results/ablation_checkpoint/taxiBJ/'

dset_dir = 'D:/Separate_factors/data/processed/taxiBJ/'
dset_name = 'TaxiBJ_st'

random_seed = 42
max_epoch = 500
batch_size = 128

# # full features model args
# model_args_full = [1000, 2000, 1000, 1024]
# learning_rate_full = 0.0005

# # info features model args
# model_args_info = [1000, 2000, 1000, 1024]
# learning_rate_info = 0.0001

# full features model args
model_args_full = [1000, 2000, 2000, 1024]
learning_rate_full = 0.0005

# info features model args
model_args_info = [1000, 2000, 2000, 1024]
learning_rate_info = 0.0003


csv_folder_path = "D:/Separate_factors/repo/results/full_info/taxiBJ/"

