import os
import logging

import gin
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from .utils import *
from ..utils import mkdirs
from .baseSolver import BaseSolver, ModelFilter

from ..models.vaes.others.baseOther import BaseDiscriminator
from ..models.vaes import *
from ..dataParser.trainingDataset.utils import *

@gin.configurable('StVAESolver', blacklist=['mode', 'log_dir', 'ckpt_dir'])
class StVAESolver(BaseSolver):
    # The super class for all solver class, describe all essential functions.
    def __init__(self, mode, log_dir, ckpt_dir, 
                log_flag=gin.REQUIRED, ckpt_save_iter=gin.REQUIRED, 
                ckpt_load=gin.REQUIRED, print_iter=gin.REQUIRED, 
                gamma=gin.REQUIRED, ckptname=gin.REQUIRED, 
                VAE=gin.REQUIRED, D=gin.REQUIRED, 
                lr_VAE=gin.REQUIRED, lr_D=gin.REQUIRED, 
                beta1_VAE=gin.REQUIRED, beta2_VAE=gin.REQUIRED, 
                beta1_D=gin.REQUIRED, beta2_D=gin.REQUIRED):
        super(StVAESolver, self).__init__()
        self.model_name = 'stVAE'
        self.log_dir = log_dir
        self.log_flag = log_flag
        self.ckpt_dir  = ckpt_dir + 'stVAE/'
        self.ckpt_save_iter  = ckpt_save_iter 
        self.ckpt_load  = ckpt_load 
        self.print_iter = print_iter
        self.gamma = gamma
        self.mode = mode
        self.ckptname = ckptname

        mkdirs(self.ckpt_dir)
        if self.log_flag:
            self.logger = logging.getLogger('rootlogger')
            self.model_handler = logging.FileHandler(self.log_dir+'{}_train_log.log'.format(self.model_name))
            self.model_handler.setLevel(logging.DEBUG)
            self.model_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.log_filter = ModelFilter(self.model_name)
            self.model_handler.addFilter(self.log_filter)
            self.logger.addHandler(self.model_handler)
        
        if not self.ckpt_load and self.log_flag:
            self.logger.info('{}: Solver initialising.'.format(self.model_name))

        self.data_loader = get_loader(self.model_name, self.mode)
        self.VAE = VAE().to(self.device)
        self.D = D().to(self.device)

        self.lr_VAE = lr_VAE
        self.beta1_VAE = beta1_VAE
        self.beta2_VAE = beta2_VAE

        self.lr_D = lr_D
        self.beta1_D = beta1_D
        self.beta2_D = beta2_D

        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))
        
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))
        self.nets = [self.VAE, self.D]

        # Checkpoint loading
        if ckpt_load:
            self.load_checkpoint(ckptname=ckptname, verbose=ckpt_load)

    def train(self):
        if self.log_flag:
            self.logger.info('{}: training starts.'.format(self.model_name))
        self.net_mode(train_flag=True)

        ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        progress_template = 'stVAE: [{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_tc_loss:{:.3f}'

        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                
                #x_recon, (spatial_mu, spatial_sigma), (temporal_mu, temporal_sigma), (spatial_z, temporal_z), z
                x_recon, spatial_dist, temporal_dist, (spatial_z, temporal_z), z = self.VAE(x_true1)

                vae_recon_loss = recon_loss(x_true1, x_recon)

                vae_kld_spatial, _, _ = kl_divergence(*spatial_dist)
                vae_kld_temporal, _, _ = kl_divergence(*temporal_dist)

                # mi_loss = mutual_information_regularization(spatial_z, temporal_z)
                # spatial_mi = mutual_information(spatial_z, spatial_dist)
                # temporal_mi = mutual_information(temporal_z, temporal_dist)
                


                vae_kld = vae_kld_spatial + vae_kld_temporal

                D_z = self.D(z)
                vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()


                # vae_loss = vae_recon_loss + vae_kld + self.gamma1 * vae_tc_loss + self.gamma2 * mi_loss
                # vae_loss = vae_recon_loss + vae_kld + self.gamma * vae_tc_loss + self.gamma * mi_loss

                vae_loss = vae_recon_loss + vae_kld + self.gamma * vae_tc_loss
                # vae_loss = vae_recon_loss + vae_kld + self.gamma * vae_kld

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()

                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True)

                # z_prime = self.VAE(x_true2, no_dec=True)
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_tc_loss.backward()
                self.optim_D.step()
            
                result_str = progress_template.format(self.global_iter, vae_recon_loss.item(), 
                                                        vae_kld.item(), vae_tc_loss.item(), D_tc_loss.item())
                if self.log_flag:
                    self.logger.debug(result_str)

                if self.global_iter%self.print_iter == 0:
                    self.pbar.write(result_str)

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)
            
                if self.global_iter >= self.max_iter:
                    out = True
                    break
        if self.log_flag:
            self.logger.info('{}: training finished.'.format(self.model_name))
        self.pbar.write("[Training Finished]")
        self.pbar.close()
        self.model_handler.close()
        
    def net_mode(self, train_flag):
        if not isinstance(train_flag, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train_flag:
                net.train()
            else:
                net.eval()
    
    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname=-1, verbose=True):
        if ckptname == -1:
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])
        else:
            ckptname = str(ckptname)

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
        else:
            if verbose:
                print('2')
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))

    # def clean(self):
    #     del self.D
    #     del self.VAE
    #     del self.optim_D
    #     del self.optim_VAE