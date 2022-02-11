import os
import math
import logging

import gin
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import distributions as dist

class DataGather(object):
    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg:[] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

def recon_loss(x, x_recon):
    '''
        Calculate the binary cross entropy between recon and x.
        Noted that it use 'binary_cross_entropy_with_logits' which 
        means the decoder doesn't need  a sigmoid layer.
    '''
    batch_size = x.size(0)
    assert batch_size != 0
    flat_recon = torch.flatten(x_recon, start_dim=1)
    flat_input = torch.flatten(x, start_dim=1)

    # x_recon = F.sigmoid(x_recon)
    loss = F.mse_loss(flat_input, flat_recon, size_average=False).div(batch_size)
    # loss = F.mse_loss(x, x_recon, size_average=False).div(batch_size)
    # loss = F.mse_loss(x_recon, x)
    # loss = F.l1_loss(x_recon, x, size_average=False).div(batch_size)
    return loss


def kl_divergence(mu, logvar):
    # Calculate the KL divergnece based on the mu and logvar.
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld


def permute_dims(z):
    # Dedicated for FactorVAE.
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)

def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.
    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix
    Parameters
    ----------
    batch_size: int
        number of training images in the batch
    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=False):
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx

@gin.configurable('mutual_information')
def mutual_information(latent_sample, latent_dist, n_data=gin.REQUIRED, is_mss=False):
    log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, 
                                                                            n_data, is_mss)
    # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
    mi_loss = (log_q_zCx - log_qz).mean()
    return mi_loss

def mutual_information_regularization(z_f, z_t):
    def dist_op(dist1, op):
        return dist.Normal(loc=op(dist1.loc), scale=op(dist1.scale))
    # H_t entropy
    # below shorthand for
    # z_f.loc = z_f.loc.unsqueeze(0)
    # z_f.scale = z_f.scale.unsqueeze(0)
    z_t1 = dist_op(z_t, lambda x: x.unsqueeze(1))
    z_t2 = dist_op(z_t, lambda x: x.unsqueeze(2))
    log_q_t = z_t1.log_prob(z_t2.rsample()).sum(-1)
    # 2 is important here!
    H_t = log_q_t.logsumexp(2).mean(1) - np.log(log_q_t.shape[2])

    z_f1 = dist_op(z_f, lambda x: x.unsqueeze(1))
    z_f2 = dist_op(z_f, lambda x: x.unsqueeze(2))
    log_q_f = z_f1.log_prob(z_f2.rsample()).sum(-1)

    H_f = log_q_f.logsumexp(2).mean(1) - np.log(log_q_t.shape[2])
    H_ft = (log_q_f + log_q_t).logsumexp(1).mean(1)

    mi_loss = -(H_f + H_t.mean() - H_ft.mean())
    return mi_loss