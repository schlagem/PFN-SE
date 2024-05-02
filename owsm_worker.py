from hpbandster.core.worker import Worker
import time
import torch
import torch.nn as nn
from train import train
import priors.rl_prior
import encoders
import utils
from calc_val_loss_table import val_loss_table
import math


class OSWMWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = kwargs.get('run_id', None)

    def compute(self, config_id, config, budget, working_directory):
        hps = {**config}  # sampled hps
        start_time = time.time()
        # Criterion to optimize oswm
        criterion = nn.MSELoss(reduction='none')
        num_features = 14
        train_len = 1000
        min_train_len = 500
        max_dataset_size = 1001
        epochs = int(budget)
        steps_per_epoch = 100

        train_result = train(
            # the prior is the key. It defines what we train on. You should hand over a dataloader here
            # you can convert a `get_batch` method to a dataloader with `priors.utils.get_batch_to_dataloader`
            get_batch_method=priors.rl_prior.get_batch, criterion=criterion,
            # define the transformer size
            emsize=512, nhead=4, nhid=1024, nlayers=6,
            # how to encode the x and y inputs to the transformer
            encoder_generator=encoders.Linear, y_encoder_generator=encoders.Linear,
            # these are given to the prior, which needs to know how many features we have etc
            extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
            # change the number of epochs to put more compute into a training
            # an epoch length is defined by `steps_per_epoch`
            epochs=epochs, warmup_epochs=epochs // 4, steps_per_epoch=steps_per_epoch, batch_size=4,
            # the lr is what you want to tune! usually something in [.00005,.0001,.0003,.001] works best
            lr=.00005,
            # seq_len defines the size of your datasets (including the test set)
            seq_len=max_dataset_size,
            # single_eval_pos_gen defines where to cut off between train and test set
            # a function that (randomly) returns lengths of the training set
            # the below definition, will just choose the size uniformly at random up to `max_dataset_size`
            single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(train_len, min_train_len, p=0.4))

        final_mean_loss, final_per_datasetsize_losses, trained_model, dataloader = train_result

        results = val_loss_table(trained_model, debug_truncation=False)
        cartpole_loss = (results["CartPole-v1"]["1.0"]["loss"] +
                        results["CartPole-v1"]["0.5"]["loss"] +
                        results["CartPole-v1"]["0.0"]["loss"]) / 3.
        reacher_loss = (results["Reacher-v4"]["1.0"]["loss"] +
                       results["Reacher-v4"]["0.5"]["loss"] +
                       results["Reacher-v4"]["0.0"]["loss"]) / 3.
        pendulum_loss = (results["Pendulum-v1"]["1.0"]["loss"] +
                        results["Pendulum-v1"]["0.5"]["loss"] +
                        results["Pendulum-v1"]["0.0"]["loss"]) / 3.
        simpleenvloss = (results["SimpleEnv"]["1.0"]["loss"] +
                        results["SimpleEnv"]["0.5"]["loss"] +
                        results["SimpleEnv"]["0.0"]["loss"]) / 3.

        # TODO weight different losses or exclude some
        over_all_mean_loss = (cartpole_loss + reacher_loss + pendulum_loss + simpleenvloss) / 4.
        score = over_all_mean_loss
        if math.isnan(score):
            score = 1000.
        run_time = time.time() - start_time
        info_dict = {"run_time": run_time,
                     "cartpole_loss": cartpole_loss,
                     "reacher_loss": reacher_loss,
                     "pendulum_loss": pendulum_loss,
                     "simpleenv_loss": simpleenvloss,
                     "train_loss": final_mean_loss,
                     "all_resutls": results
                     }
        return ({
            'loss': float(score),  # this is the a mandatory field to run hyperband
            'info': info_dict  # can be used for any user-defined information - also mandatory
        })