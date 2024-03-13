import torch
import numpy as np

from .prior import Batch
from utils import default_device
import gym



@torch.no_grad()
def get_batch(
        batch_size,
        seq_len,
        num_features,
        device=default_device,
        hyperparameters=None,
        **kwargs
):
    # TODO num features is equal to state -> output is then action
    # TODO requires perfect actions to learn from
    X = torch.full((seq_len, batch_size, num_features), 0.)
    Y = torch.full((seq_len, batch_size, num_features), float(-100.))  #TODO  Nan is skipped
    # TODO load precomputed transitions
    for b in range(batch_size):
        for i in range(seq_len):
            pass
    # TODO normalize -> find max value TOOD
    # Here shuffle that the masking does not only mask later values -> which leads
    # per = torch.randperm(X.shape[0])
    # X = X[per]
    # Y = Y[per]
    return Batch(x=X, y=Y, target_y=Y)

