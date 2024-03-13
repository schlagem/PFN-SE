import sys
sys.path.insert(0,'..')

import torch
from torch import nn
import numpy as np
import random

import matplotlib.pyplot as plt

from train import train
import priors.fast_gp
import priors.rl_prior
import encoders
import positional_encodings
import utils
import bar_distribution
import transformer

# There might be warnings during training, regarding efficiency and a missing GPU, if using CPU
# We do not care about these for this tutorial
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

## define prior hyperparameter
# num_features = 1
# hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6, 'fast_computations': (False,False,False)}
# let's sample from the prior to understand it a bit
# batch = priors.fast_gp.get_batch(batch_size=50, seq_len=100, num_features=num_features, hyperparameters=hps)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# maximum Dimension of observations + max dimension of action
# here dim obs 6 and dim action 2 as acrobot has 2 -> 7
num_features = 7  # 11 + 3
hps = {'test': True}
batch, x_means, x_stds, y_means, y_stds = priors.rl_prior.get_batch(batch_size=40, seq_len=1500, num_features=num_features, hyperparameters=hps)

print(x_means.shape)
print(x_stds.shape)
print(y_means)
print(y_stds)

print('Our x shape (seq_len, batch_size, num_features), sometimes we refer to seq_len as seq_len or dataset size:\n',
      f'{batch.x.shape=}')

print(f'Our y and target_y shape is (seq_len, batch_size)\n {batch.y.shape=}, {batch.target_y.shape=}')

# define a bar distribution (riemann distribution) criterion with 1000 bars
# this follow the standard criterion API (without reduction)
# For classification targets, you can equivalently use something like nn.CrossEntropyLoss(reduction='none')

# ys = priors.fast_gp.get_batch(100000, 20, num_features, hyperparameters=hps).target_y
# we define our bar distribution adaptively with respect to the above sample of target ys from our prior
# criterion = bar_distribution.FullSupportBarDistribution(bar_distribution.get_bucket_limits(num_outputs=1000, ys=ys))
#criterion = nn.CrossEntropyLoss(reduction='none') # if your prior is a classification prior (i.e. `y_targets` are ints)
criterion = nn.MSELoss(reduction='none')

# number of data points provided at train time
train_len = 1000

max_dataset_size = 1500
epochs = 15 # 50
hps = {'test': False}
train_result = train(# the prior is the key. It defines what we train on. You should hand over a dataloader here
                     # you can convert a `get_batch` method to a dataloader with `priors.utils.get_batch_to_dataloader`
                     get_batch_method=priors.rl_prior.get_batch, criterion=criterion,
                     # define the transformer size
                     emsize=512, nhead=4, nhid=1024, nlayers=6,
                     # emsize=512, nhead=8, nhid=1024, nlayers=12,
                     #emsize=800, nhead=8, nhid=1024, nlayers=12,
                     # emsize=1024, nhead=8, nhid=2048, nlayers=8,
                     # how to encode the x and y inputs to the transformer
                     encoder_generator=encoders.Linear, y_encoder_generator=encoders.Linear,
                     # these are given to the prior, which needs to know how many features we have etc
                     extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
                     # change the number of epochs to put more compute into a training
                     # an epoch length is defined by `steps_per_epoch`
                     # the below means we do 10 epochs, with 100 batches per epoch and 4 datasets per batch
                     # that means we look at 10*1000*4 = 4000 datasets. Considerably less than in the demo.
                     epochs=epochs, warmup_epochs=epochs//4, steps_per_epoch=100, batch_size=4, # steps per epoch 100
                     # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
                     # the lr interacts heavily with `batch_size` (smaller `batch_size` -> smaller best `lr`)
                     lr=.00005,
                     # seq_len defines the size of your datasets (including the test set)
                     seq_len=max_dataset_size,
                     # single_eval_pos_gen defines where to cut off between train and test set
                     # a function that (randomly) returns lengths of the training set
                     # the below definition, will just choose the size uniformly at random up to `max_dataset_size`
                     single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(train_len + 1, min_len=train_len))

final_mean_loss, final_per_datasetsize_losses, trained_model, dataloader = train_result


torch.save(trained_model.state_dict(), "trained_models/prior_smaller_NNs.pt")

train_x = batch.x[:train_len]
train_y = batch.y[:train_len]
test_x = batch.x[:]

with torch.no_grad():
    logits = trained_model(train_x, train_y, test_x)



# the model has the criterion still attached (it is the same though, as our criterion above)
# the criterion has a lot of handy function to use these logits
y = (logits * y_stds) + y_means
x = (test_x * x_stds) + x_means
y_target = (batch.y * y_stds) + y_means

with torch.no_grad():
    torch.set_printoptions(sci_mode=False)
    print("Final test loss:", torch.nn.functional.mse_loss(y, y_target))
    print("Testloss per axis:", torch.nn.functional.mse_loss(y, y_target, reduction="none").mean(axis=(0, 1)))



fig, axs = plt.subplots(2, 3)
fig.tight_layout()

# This Scatter plots the data points to see over or under estimation
for i, ax in enumerate(axs.reshape(-1)):
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


# This plots Loss for each dimension
fig, axs = plt.subplots(1, num_features)
fig.tight_layout()

for i, ax in enumerate(axs.reshape(-1)):
    axx = x[:, :, i].view(-1)
    train_points = x[:train_len, :, i].view(-1)
    axy = torch.nn.functional.mse_loss(y[:, :, i], y_target[:, :, i], reduction="none").view(-1)
    ind = torch.argsort(axx)
    ax.scatter(torch.zeros(axx.shape), axx, s=.5, c="green")
    ax.scatter(torch.zeros(train_points.shape), train_points, s=.5, c="red")
    ax.fill_betweenx(axx[ind], 0-axy[ind], 0+axy[ind], alpha=0.1, color="green")
    ax.set_ylabel("dimension " + str(i))

plt.show()
