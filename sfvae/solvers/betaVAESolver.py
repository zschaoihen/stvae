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
from ..models.vaes.betaVAE import BetaVAE
from ..dataParser.trainingDataset.utils import *

@gin.configurable('BetaVAESolver', blacklist=['mode', 'log_dir', 'ckpt_dir'])
class BetaVAESolver(BaseSolver):
    # The super class for all solver class, describe all essential functions.
    def __init__(self, mode, log_dir, ckpt_dir, 
                log_flag=gin.REQUIRED, ckpt_save_iter=gin.REQUIRED, 
                ckpt_load=gin.REQUIRED, print_iter=gin.REQUIRED, 
                objective=gin.REQUIRED, gamma=gin.REQUIRED, 
                C_max=gin.REQUIRED, C_stop_iter=gin.REQUIRED, 
                ckptname=gin.REQUIRED, 
                VAE=gin.REQUIRED, D=gin.REQUIRED, 
                lr_VAE=gin.REQUIRED, lr_D=gin.REQUIRED, 
                beta1_VAE=gin.REQUIRED, beta2_VAE=gin.REQUIRED, 
                beta1_D=gin.REQUIRED, beta2_D=gin.REQUIRED):
        super(BetaVAESolver, self).__init__(cfg)
        self.model_name = 'betaVAE'

        self.log_dir = log_dir
        self.log_flag = log_flag
        self.ckpt_dir  = ckpt_dir + 'betaVAE/'
        self.ckpt_save_iter  = ckpt_save_iter 
        self.ckpt_load  = ckpt_load 
        self.print_iter = print_iter

        self.objective = objective
        self.gamma = gamma
        self.C_max = C_max
        self.C_stop_iter = C_stop_iter

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
        

        # Get Data loader
        self.data_loader = get_loader(self.model_name, self.mode)
        self.VAE = VAE().to(self.device)

        # Networks parameter readin
        self.encoder_layer_dims = json.loads(cfg[self.model_name]['encoder_layer_dims'])
        self.decoder_layer_dims = json.loads(cfg[self.model_name]['decoder_layer_dims'])

        self.lr_VAE = lr_VAE
        self.beta1_VAE = beta1_VAE
        self.beta2_VAE = beta2_VAE

        if self.mode[0] == 'stgcn':
            torch.backends.cudnn.enabled = False

        self.optim = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))
        self.net = self.VAE
        
        # Checkpoint loading
        if ckpt_load:
            self.load_checkpoint(verbose=ckpt_load)

    def train(self): 
        if self.log_flag:
            self.logger.info('{}: training starts.'.format(self.model_name))
        self.net_mode(train_flag=True)

        self.C_max = Variable(torch.FloatTensor([self.C_max]).to(self.device))

        progress_template = 'betaVAE: [{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'

        out = False
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                self.pbar.update(1)

                if self.mode[0] != 'stgcn':
                    x = Variable(x.to(self.device))
                    x_recon, mu, logvar, z = self.net(x)
                else:
                    x, adj_mx = x
                    adj_mx = Variable(adj_mx[0].squeeze()).to(self.device)
                    x = Variable(x.to(self.device))
                    mu, logvar = self.net.encode_((adj_mx, x))
                    z = self.net.reparameterize(mu, logvar)
                    x_recon = self.net.decode_((adj_mx, z))
                    z = z.squeeze()

                # x = Variable(x.to(self.device))
                # x_recon, mu, logvar, z = self.net(x)
                vae_recon_loss = recon_loss(x, x_recon)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = vae_recon_loss + self.gamma*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = vae_recon_loss + self.gamma*(total_kld-C).abs()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                result_str = progress_template.format(self.global_iter, vae_recon_loss.item(), 
                                                        total_kld.item(), mean_kld.item())
                
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

    def train_iter(self, x):
        self.net_mode(train_flag=True)
        
        x = Variable(x.to(self.device))
        self.C_max = Variable(torch.FloatTensor([self.C_max]).to(self.device))
        x_recon, mu, logvar, z = self.net(x)
        vae_recon_loss = recon_loss(x, x_recon)
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        if self.objective == 'H':
            beta_vae_loss = vae_recon_loss + self.gamma*total_kld
        elif self.objective == 'B':
            C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            beta_vae_loss = vae_recon_loss + self.gamma*(total_kld-C).abs()

        self.optim.zero_grad()
        beta_vae_loss.backward()
        self.optim.step()

    def net_mode(self, train_flag):
        if not isinstance(train_flag, bool):
            raise ValueError('Only bool type is supported. True|False')

        if train_flag:
            self.net.train()
        else:
            self.net.eval()
    
    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
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

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            verbose_text = "betaVAE: => loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter)
        else:
            verbose_text = "betaVAE: => no checkpoint found at '{}'".format(filepath)