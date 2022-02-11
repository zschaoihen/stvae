import os
import json 
import argparse
import logging
import logging.handlers
import copy
import gin

import numpy as np
import torch

from .utils import str2bool, mkdirs
from .exps import *
from .solvers import *

@gin.configurable('traversal')
def single_traversal(seed, exp_models, gamma, log_dir, ckpt_dir, traversal_dir):
    temp_log_dir = log_dir + '{}_{}/'.format(seed, gamma)
    temp_ckpt_dir = ckpt_dir + '{}_{}/'.format(seed, gamma)
    mkdirs(temp_log_dir)
    mkdirs(temp_ckpt_dir)

    temp_tra_dir = traversal_dir + '{}_{}/'.format(seed, gamma)
    mkdirs(temp_tra_dir)

    for model in exp_models:
        temp_module_name = "{}Solver".format(model)
        temp_func_name = temp_module_name[0].upper() + temp_module_name[1:]
        if temp_module_name in globals():
            model_solver = globals()[temp_module_name].__dict__[temp_func_name](mode, temp_log_dir, temp_ckpt_dir)
            assert model_solver.ckpt_load
            temp_VAE = model_solver.VAE

        else:
            print("no such solver: {} solver found in this package".format(model))


def traversal():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    result_path = base_path + 'results/{}/'.format(run_number)
    ckpt_dir = result_path + 'models/'
    log_dir = result_path + 'log/'
    traversal_dir = result_path + 'visual/traversal'

    mkdirs(traversal_dir)

    for seed in seeds:
        for gamma in gammas:
            single_traversal(seed, exp_models, gamma, log_dir, ckpt_dir, traversal_dir) 