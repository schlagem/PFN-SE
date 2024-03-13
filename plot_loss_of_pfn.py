import math
from typing import Optional, Union, List

import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym.core import RenderFrame
import numpy as np

from priors.prior import Batch
import priors.rl_prior
from train import build_model
import encoders

import priors.rl_prior
import time
from os import path


num_features = 7
# building Transformer model and loading weights
criterion = nn.MSELoss(reduction='none')
# TODO test batch?
hps = {'test': True}
pfn = build_model(
    criterion=criterion,
    encoder_generator=encoders.Linear,
    test_batch=None,
    n_out=7,
    emsize=512, nhead=4, nhid=1024, nlayers=6,
    # emsize=512, nhead=8, nhid=1024, nlayers=12,
    # emsize=800, nhead=8, nhid=1024, nlayers=12,
    # emsize=1024, nhid=2048, nlayers=8, nhead=8,
    seq_len=1500,
    y_encoder_generator=encoders.Linear,
    decoder_dict={},
    extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
)
print(
    f"Using a Transformer with {sum(p.numel() for p in pfn.parameters()) / 1000 / 1000:.{2}f} M parameters"
)


# pfn.load_state_dict(torch.load("trained_models/se_transformer_retrained_shuffled_noise_added_random_env_large.pt"))
# pfn.load_state_dict(torch.load("trained_models/se_pfn_transfer.pt"))
# pfn.load_state_dict(torch.load("trained_models/semi_working_prioir_small.pt"))
# pfn.load_state_dict(torch.load("trained_models/se_pfn_transfer_new_to_not_overwrite.pt"))
# pfn.load_state_dict(torch.load("trained_models/prior_changes_small.pt"))
# pfn.load_state_dict(torch.load("trained_models/prior_larger.pt"))
pfn.load_state_dict(torch.load("trained_models/prior_retrain_working.pt"))
pfn.eval()

batch, x_means, x_stds, y_means, y_stds = priors.rl_prior.get_batch(batch_size=10, seq_len=1500, num_features=num_features, hyperparameters=hps)

print(x_means)
print(x_stds)

train_len = 1000

train_x = batch.x[:train_len]
train_y = batch.y[:train_len]
test_x = batch.x[:]

start = time.time()
with torch.no_grad():
    logits = pfn(train_x, train_y, test_x)
print("Time of forward: ", time.time() - start)

# the model has the criterion still attached (it is the same though, as our criterion above)
# the criterion has a lot of handy function to use these logits
y = (logits * y_stds) + y_means
x = (test_x * x_stds) + x_means
y_target = (batch.y * y_stds) + y_means

# y[1000:, :, :6] = torch.round(y[1000:, :, :6])



with torch.no_grad():
    torch.set_printoptions(sci_mode=False)
    print("Mean Final test loss:", torch.nn.functional.mse_loss(torch.mean(y[1000:, :, :], dim=1), torch.mean(y_target[1000:, :, :], dim=1)))
    print("Mean Testloss per axis:", torch.nn.functional.mse_loss(torch.mean(y[1000:, :, :], dim=1), torch.mean(y_target[1000:, :, :], dim=1), reduction="none").mean(axis=(0)))
    print("Final test loss:", torch.nn.functional.mse_loss(y[1000:, :, :], y_target[1000:, :, :]))
    print("Testloss per axis:", torch.nn.functional.mse_loss(y[1000:, :, :], y_target[1000:, :, :], reduction="none").mean(axis=(0, 1)))


"""
fig, axs = plt.subplots(1, 1)
fig.tight_layout()

# This Scatter plots the data points to see over or under estimation
for i, ax in enumerate(axs.reshape(-1)):
    plt.cla()
    plt.clf()
    t_max = torch.max(torch.abs(x[train_len:, :, i]))
    p_max = torch.max(torch.abs(y[train_len:, :, i]))
    print(f"Dimension {i} - Max target - {t_max} - Max predictions {p_max}")
    # Center line
    ax.plot(np.arange(-10, 10, 1), np.arange(-10, 10, 1), alpha=0.1)
    lim = max(p_max, t_max) + 0.1
    ax.set_ylim([-lim, lim])
    ax.set_xlim([-lim, lim])
    ax.scatter(x[train_len:, :, i], y[train_len:, :, i], s=3, c="tab:orange")
    ax.set_ylabel("target")
    ax.set_ylabel("prediction")
    plt.show()
"""
print(x[train_len:, :, :].shape)

# This Scatter plots the data points to see over or under estimation
for i in range(num_features):
    plt.cla()
    plt.clf()
    t_max = torch.max(torch.abs(y_target[train_len:, :, i]))
    p_max = torch.max(torch.abs(y[train_len:, :, i]))
    print(f"Dimension {i} - Max target - {t_max} - Max predictions {p_max}")
    # Center line
    plt.plot(np.arange(-10, 10, 1), np.arange(-10, 10, 1), alpha=0.1)
    lim = max(p_max, t_max) + 0.1
    plt.ylim([-lim, lim])
    plt.xlim([-lim, lim])
    for b in range(y_target.shape[1]):
        plt.scatter(y_target[train_len:, b, i], y[train_len:, b, i], s=3., c="tab:orange")
    # plt.scatter(torch.mean(y_target[train_len:, :, i], dim=1), torch.mean(y[train_len:, :, i], dim=1),  s=3)
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.show()


