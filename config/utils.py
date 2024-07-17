import numpy as np
import tensorflow as tf
import torch
from torch import nn as nn



def truncated_normal(size, std):
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = False

    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session(config=cfg)
    val = sess.run(tf.compat.v1.truncated_normal(shape=size, stddev=std))

    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features), std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b