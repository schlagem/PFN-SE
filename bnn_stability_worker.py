import torch
import numpy as np
from priors.rl_prior import BNN, SinActivation, NoOpActivation
from hpbandster.core.worker import Worker
import time
from train import train
import priors.rl_prior
import utils
from calc_val_loss_table import val_loss_table
import math
from decoder import *



dict_act = {"relu": torch.nn.ReLU,
            "leaky": torch.nn.LeakyReLU,
            "elu": torch.nn.ELU,
            "sin": SinActivation,
            "identity": NoOpActivation,
            "tanh": torch.nn.Tanh,

            }


def generate_bnn(in_size, out_size, hps):
    depth = hps["depth"]
    width = hps["width"]
    additive_noise_std = hps["noise"]
    init_std = hps["init"]  # TNLU(10., 0.01, 0.0, to_round=False)

    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=init_std)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=init_std)

    use_bias = hps["bias"]
    use_res_connection = hps["res"]
    act_funct = dict_act[hps["act"]]
    dropout = hps["dropout"]
    if dropout:
        dropout_p = hps["p_drop"]
    else:
        dropout_p = 0.
    # 0.9 * np.random.beta(np.random.uniform(0.1, 5.0), np.random.uniform(0.1, 5.0))
    bnn_model = BNN(in_size, out_size, depth, width, use_bias, use_res_connection, act_funct, dropout, dropout_p,
                    additive_noise_std)
    bnn_model.apply(weight_init)
    return bnn_model


def get_bnn_train_batch(seq_len, batch_size, num_features, hyperparameters):
    # generate input data
    X = torch.rand((seq_len, batch_size, num_features))
    Y = torch.Tensor()
    for b in range(batch_size):
        # sample random state dim and action dim
        state_dim = np.random.randint(3, 12)
        action_dim = np.random.randint(1, 4)

        # zero out not used action dims
        X[:, b, state_dim:-3] = 0
        X[:, b, num_features-3+action_dim:] = 0

        # sample BNN for state dym
        state_dynamics_bnn = generate_bnn(state_dim + action_dim, state_dim, hyperparameters)
        # forward state dym BNN
        # TODO improve representation of action
        next_state = state_dynamics_bnn(
            torch.cat((X[:, b:b+1, :state_dim], X[:, b:b+1, num_features - 3:num_features - 3 + action_dim]), dim=2))

        # sample BNN for reward
        reward_dynamics_bnn = generate_bnn(2 * state_dim + action_dim, 1, hyperparameters)

        # forward BNN for reward
        reward = reward_dynamics_bnn(
            torch.cat((X[:, b:b + 1, :state_dim], X[:, b:b + 1, num_features - 3:num_features - 3 + action_dim],
                       next_state), dim=2))


        final_total_dym = torch.cat(
            (next_state, torch.zeros((seq_len, 1, num_features - 1 - state_dim)), reward), dim=2)

        # Fill Output
        Y = torch.cat((Y, final_total_dym), dim=1)

        # shuffle zero dims
        state_per_dims = torch.randperm(num_features-3)
        X[:, b:b+1, :num_features-3] = X[:, b:b+1, :num_features-3][:, :, state_per_dims]

        Y[:, b:b+1, :num_features-3] = Y[:, b:b+1, :num_features-3][:, :, state_per_dims]

        action_per_dim = torch.randperm(3)
        X[:, b:b+1, num_features-3:] = X[:, b:b+1, num_features-3:][:, :, action_per_dim]

    # 0 mean 1 variacne Normalize
    x_means = torch.mean(X, dim=0)
    x_stds = torch.std(X, dim=0)
    X = torch.nan_to_num((X - x_means) / x_stds, nan=0)

    # min max scaling
    """
    y_min = Y.min(dim=0, keepdim=True).values.min(dim=0, keepdim=True).values
    y_max = Y.max(dim=0, keepdim=True).values.max(dim=0, keepdim=True).values
    Y = torch.nan_to_num((Y - y_min)/(y_max - y_min))
    """

    # 0 mean 1 variacne Normalize
    y_means = torch.mean(Y, dim=0)
    y_stds = torch.std(Y, dim=0)
    Y = torch.nan_to_num((Y - y_means) / y_stds, nan=0)
    return X, Y


class BNN_worker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = kwargs.get('run_id', None)

    def compute(self, config_id, config, budget, working_directory):
        hps = {**config}
        nan_batches = 0
        for i in range(int(budget)):
            X, Y = get_bnn_train_batch(1001, 4, 14, hps)
            # print(torch.mean(Y, dim=0))
            if Y.max() > 340282346638528. or torch.isinf(torch.mean(Y, dim=0)).any():
                nan_batches += 1
        score = 1 - (nan_batches / float(budget))
        info_dict = {}
        return ({
            'loss': float(score),  # this is the a mandatory field to run hyperband
            'info': info_dict  # can be used for any user-defined information - also mandatory
        })