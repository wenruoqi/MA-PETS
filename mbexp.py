from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from dotmap import DotMap
from MBExperiment import MBExperiment
from MPC import MPC
from config import create_config
import torch
import numpy as np
import random
import tensorflow as tf
import sys

root_path = os.path.abspath(__file__) 
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)

def set_global_seeds(seed):
    torch.manual_seed(seed) 
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tf.compat.v1.random.set_random_seed(seed) 


def mbexp(env, ctrl_type, ctrl_args, overrides, logdir):
    set_global_seeds(0)

    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir) 

    assert ctrl_type == 'MPC' 

    cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)  
    exp = MBExperiment(cfg.exp_cfg)
    exp.run_experiment()