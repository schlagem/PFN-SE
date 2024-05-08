import sys
sys.path.insert(0, '..')

import torch
from torch import nn
import numpy as np
import random
from train import train
import priors.rl_prior
import encoders
import utils
from decoder import DecoderModel

# There might be warnings during training, regarding efficiency and a missing GPU, if using CPU
# We do not care about these for this tutorial
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)

# maximum Dimension of observations + max dimension of action
num_features = 14
hps = {"env_name": "MomentumEnv", "num_hidden": 1, "relu": False, "sigmoid": False, "sin": True,
       "state_offset": 3.2802608490289904, "state_scale": 18.147661409701062, "tanh": True, "test": False,
       "use_bias": False, "use_dropout": False, "use_layer_norm": True, "use_res_connection": True, "width_hidden": 16,
       "no_norm": False}

hps = {}

criterion = nn.MSELoss(reduction='none')

# number of data points provided at train time
train_len = 1000
min_train_len = 500

max_dataset_size = 1001
epochs = 50
train_result = train(# the prior is the key. It defines what we train on. You should hand over a dataloader here
                     # you can convert a `get_batch` method to a dataloader with `priors.utils.get_batch_to_dataloader`
                     get_batch_method=priors.rl_prior.get_bnn_batch, criterion=criterion,
                     # define the transformer size
                     # emsize=1024, nhead=16, nhid=2048, nlayers=10,
                     emsize=512, nhead=4, nhid=1024, nlayers=6,
                     # how to encode the x and y inputs to the transformer
                     encoder_generator=encoders.Linear,
                     y_encoder_generator=encoders.Linear,
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
                     #single_eval_pos_gen=utils.get_uniform_single_eval_pos_sampler(train_len + 1, min_len=train_len))
                     single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(train_len, min_train_len, p=0.4))

final_mean_loss, final_per_datasetsize_losses, trained_model, dataloader = train_result


torch.save(trained_model.state_dict(), "trained_models/BNN_testing.pt")

