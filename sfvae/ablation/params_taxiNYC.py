config_name = "run_taxiNYC"
scaler_path = "D:/Separate_factors/data/processed/taxiNYC/TaxiNYC_scaler.pkl"

# double size
gin_file_path = './configs/{}/main_setup.gin'.format(config_name)
run_number_list = [403, 406]
single_input = 288

# # single size
# gin_file_path = './configs/{}/main_setup_1.gin'.format(config_name)
# run_number_list = [420, 421, 422, 423]
# single_input = 144

#curent
base_path = './'
model_name = "stVAE"

model_index = -1
gamma_list = [0.4, 1.6]
seed_list = [35, 42]
mode = ['spatio_temporal', 'revised']

MAX_VALUE = 1283
train_portion = 0.7
valid_portion = 0.1

result_dir = 'D:/Separate_factors/repo/results/ablation_checkpoint/taxiNYC/'

dset_dir = 'D:/Separate_factors/data/processed/taxiNYC/'
dset_name = 'TaxiNYC_st'

random_seed = 42
max_epoch = 500
batch_size = 128

# # full features model args
# model_args_full = [500, 1000, 2000, 1000, 500, 200]
# learning_rate_full = 0.0008

# # info features model args
# model_args_info = [500, 1000, 2000, 1000, 200]
# learning_rate_info = 0.0001

# # full features model args
# model_args_full = [500, 1000, 1000, 500, 200]
# learning_rate_full = 0.0005

# # info features model args
# model_args_info = [500, 1000, 1000, 200]
# learning_rate_info = 0.0003

# # full features model args
# model_args_full = [500, 1200, 500, 200]
# learning_rate_full = 0.0005

# # info features model args
# model_args_info = [500, 1200, 500, 200]
# learning_rate_info = 0.0003

# full features model args
model_args_full = [200, 500, 1200, 500, 200]
learning_rate_full = 0.0005

# info features model args
model_args_info = [200, 500, 1200, 500, 200]
learning_rate_info = 0.0003

csv_folder_path = "D:/Separate_factors/repo/results/full_info/taxiNYC/"
