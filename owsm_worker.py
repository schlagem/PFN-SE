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


def cat_encoder_generator_generator(hps, target):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNCatClass(nn.Module):
        def __init__(self, num_features, emsize, replace_nan_by_zero=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_features = num_features
            self.emsize = emsize
            self.replace_nan_by_zero = replace_nan_by_zero
            self.residual_flag = hps["encoder_res_connection"]
            self.target = target
            if self.target:
                self.part_two_in = 1
            else:
                self.part_two_in = 3

            num_hidden = hps["encoder_depth"]
            width_hidden = hps["encoder_width"]
            use_bias = hps["encoder_use_bias"]

            # First Part state or next state
            self.in_lin_1 = torch.nn.Linear(11, width_hidden, bias=use_bias)
            self.in_act_1 = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_1 = nn.ModuleList(layer_list)
            self.out_lin_1 = torch.nn.Linear(width_hidden, emsize//4, bias=use_bias)

            # Second Part action or reward
            self.in_lin_2 = torch.nn.Linear(self.part_two_in, width_hidden, bias=use_bias)
            self.in_act_2 = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_2 = nn.ModuleList(layer_list)
            self.out_lin_2 = torch.nn.Linear(width_hidden, emsize//4, bias=use_bias)

        def forward(self, x):
            if self.replace_nan_by_zero:
                x = torch.nan_to_num(x, nan=0.0)

            residual = self.in_lin_1(x[:, :, :11])
            out = self.in_act_1(residual)
            for layer in self.layer_list_1:
                out = layer(out) + self.residual_flag * residual
            out_1 = self.out_lin_1(out)

            residual = self.in_lin_2(x[:, :, self.num_features-self.part_two_in:])
            out = self.in_act_2(residual)
            for layer in self.layer_list_2:
                out = layer(out) + self.residual_flag * residual
            out_2 = self.out_lin_2(out)
            zero_pad = torch.full((x.shape[0], x.shape[1], self.emsize//2), 0.)
            if self.target:
                return torch.cat((zero_pad, out_1, out_2), dim=2)
            else:
                return torch.cat((out_1, out_2, zero_pad), dim=2)

        def __setstate__(self, state):
            super().__setstate__(state)
            self.__dict__.setdefault("replace_nan_by_zero", True)

    return NNCatClass


def mlp_encoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNClass(nn.Module):
        def __init__(self, num_features, emsize, replace_nan_by_zero=False, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_features = num_features
            self.emsize = emsize
            self.replace_nan_by_zero = replace_nan_by_zero
            num_hidden = hps["encoder_depth"]
            width_hidden = hps["encoder_width"]
            use_bias = hps["encoder_use_bias"]

            self.residual_flag = hps["encoder_res_connection"]

            self.in_lin = torch.nn.Linear(num_features, width_hidden, bias=use_bias)
            self.in_act = activation_dict[hps["encoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["encoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list = nn.ModuleList(layer_list)
            self.out_lin = torch.nn.Linear(width_hidden, emsize, bias=use_bias)

        def forward(self, x):
            if self.replace_nan_by_zero:
                x = torch.nan_to_num(x, nan=0.0)
            residual = self.in_lin(x)
            out = self.in_act(residual)
            for layer in self.layer_list:
                out = layer(out) + self.residual_flag * residual
            return self.out_lin(out)

        def __setstate__(self, state):
            super().__setstate__(state)
            self.__dict__.setdefault("replace_nan_by_zero", True)

    return NNClass


def mlp_decoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNDecClass(nn.Module):
        def __init__(self, ninp, nhid, nout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            num_hidden = hps["decoder_depth"]
            width_hidden = hps["decoder_width"]
            use_bias = hps["decoder_use_bias"]

            self.residual_flag = hps["decoder_res_connection"]

            self.in_lin = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list = nn.ModuleList(layer_list)
            self.out_lin = torch.nn.Linear(width_hidden, nout, bias=use_bias)

        def forward(self, x):
            residual = self.in_lin(x)
            out = self.in_act(residual)
            for layer in self.layer_list:
                out = layer(out) + self.residual_flag * residual
            return self.out_lin(out)

    return NNDecClass


def cat_decoder_generator_generator(hps):
    activation_dict = {"relu": torch.nn.ReLU,
                       "sigmoid": torch.nn.Sigmoid,
                       "gelu": torch.nn.GELU}

    class NNCatDecClass(nn.Module):
        def __init__(self, ninp, nhid, nout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            num_hidden = hps["decoder_depth"]
            width_hidden = hps["decoder_width"]
            use_bias = hps["decoder_use_bias"]

            self.residual_flag = hps["decoder_res_connection"]

            # Decodes state
            self.in_lin_1 = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act_1 = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_1 = nn.ModuleList(layer_list)
            self.out_lin_1 = torch.nn.Linear(width_hidden, nout-3, bias=use_bias)

            # Decodes Reward
            self.in_lin_2 = torch.nn.Linear(ninp, width_hidden, bias=use_bias)
            self.in_act_2 = activation_dict[hps["decoder_activation"]]()
            layer_list = []
            for i in range(num_hidden):
                seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias),
                            activation_dict[hps["decoder_activation"]]()]
                layer_list.append(torch.nn.Sequential(*seq_list))
            self.layer_list_2 = nn.ModuleList(layer_list)
            self.out_lin_2 = torch.nn.Linear(width_hidden, 1, bias=use_bias)

        def forward(self, x):
            residual = self.in_lin_1(x)
            out = self.in_act_1(residual)
            for layer in self.layer_list_1:
                out = layer(out) + self.residual_flag * residual
            out_1 = self.out_lin_1(out)

            residual = self.in_lin_2(x)
            out = self.in_act_2(residual)
            for layer in self.layer_list_2:
                out = layer(out) + self.residual_flag * residual
            out_2 = self.out_lin_1(out)

            zero_shape = (x.shape[0], x.shape[1], 2)
            zero_padding = torch.full(zero_shape, 0.)

            return torch.cat((out_1, zero_padding, out_2), dim=2)

    return NNCatDecClass


class OSWMWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = kwargs.get('run_id', None)

    def compute(self, config_id, config, budget, working_directory):
        fixed_hps = {"num_hidden": 1,
                     "width_hidden": 16,
                     "test": False}

        hps = {**config, **fixed_hps}
        start_time = time.time()
        # Criterion to optimize oswm
        criterion = nn.MSELoss(reduction='none')
        num_features = 14
        train_len = 1000
        min_train_len = 500
        max_dataset_size = 1001
        epochs = int(budget)
        steps_per_epoch = 100

        if hps["encoder_type"] == "mlp":
            gen_x = mlp_encoder_generator_generator(hps)
            gen_y = gen_x
        elif hps["encoder_type"] == "cat":
            gen_x = cat_encoder_generator_generator(hps, target=False)
            gen_y = cat_encoder_generator_generator(hps, target=True)

        if hps["encoder_type"] == "mlp":
            dec_model = mlp_decoder_generator_generator(hps)
        elif hps["decoder_type"] == "cat":
            dec_model = cat_decoder_generator_generator(hps)

        try:
            decoder_dict = {"standard": (dec_model, 14)}
            train_result = train(
                # the prior is the key. It defines what we train on. You should hand over a dataloader here
                # you can convert a `get_batch` method to a dataloader with `priors.utils.get_batch_to_dataloader`
                get_batch_method=priors.rl_prior.get_batch, criterion=criterion,
                # define the transformer size
                emsize=512, nhead=4, nhid=1024, nlayers=6,
                # how to encode the x and y inputs to the transformer
                encoder_generator=gen_x, y_encoder_generator=gen_y,
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
                single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(train_len, min_train_len, p=0.4),
                decoder_dict=decoder_dict)

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

            over_all_mean_loss = (cartpole_loss + reacher_loss + pendulum_loss + simpleenvloss) / 4.
            score = over_all_mean_loss
        except:
            score = 1000.
            results = None
            cartpole_loss = None
            pendulum_loss = None
            simpleenvloss = None
            final_mean_loss = None
            reacher_loss = None

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