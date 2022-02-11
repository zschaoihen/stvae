import gin

def get_eval_dataset_name(dset_class_name):
    if dset_class_name == 'taxiBJ':
        return stRasterDataset.TaxiBJDataset
    elif dset_class_name == 'melbPed':
        return stGraphDataset.MelbPedDataset
    elif dset_class_name == 'bikeNYC':
        return stGraphDataset.BikeNYCDataset

@gin.configurable('get_eval_dataset', blacklist=['mode', 'eval_flag'])
def get_eval_dataset(mode, eval_flag=False,
                    dset_class_name=gin.REQUIRED, dset_dir=gin.REQUIRED, 
                    dset_name=gin.REQUIRED):
    dset = get_eval_dataset_name(dset_class_name)

    mode = mode
    
    if mode[0] == 'series':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['y'])

    elif mode[0] == 'spatio_temporal':
        data = np.load(dset_dir + dset_name + '.npz')
        train_dset = dset(data['X'], data['y'])
    
    elif mode[0] == 'stgcn':
        # eval_flag = cfg['model_general'].getboolean('eval')
        data = np.load(dset_dir + dset_name + '.npz')
        if eval_flag:
            train_dset = dset(data['X'], data['y_flat'], data['adj_mx'])
        else:
            train_dset = dset(data['X'], data['y_flat'], data['adj_mx'], y_mode='down')

    return train_dset