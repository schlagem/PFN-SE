import unittest

import torch

import encoders
import priors
import utils
import bar_distribution
from train import train
from priors import hebo_prior, input_warping
import priors

get_batch = priors.utils.get_batch_sequence(
    hebo_prior.get_batch,
    input_warping.get_batch,
    priors.utils.sample_num_feaetures_get_batch,
)


class TestHeboTraining(unittest.TestCase):
    def test_main(self):
        c = {
            "get_batch_method": get_batch,
            "encoder_generator": encoders.get_normalized_uniform_encoder(
                encoders.get_variable_num_features_encoder(encoders.Linear)
            ),
            "emsize": 64,
            "nhead": 4,
            "warmup_epochs": 2,
            "y_encoder_generator": encoders.Linear,
            "batch_size": 16,
            "scheduler": utils.get_cosine_schedule_with_warmup,
            "extra_prior_kwargs_dict": {
                "num_features": 18,
                "hyperparameters": {
                    "input_warping_c1_std": 0.972239843636276,
                    "input_warping_c0_std": 0.5939053502155272,
                    "lengthscale_concentration": 1.2106559584074301,
                    "lengthscale_rate": 1.5212245992840594,
                    "outputscale_concentration": 0.8452312502679863,
                    "outputscale_rate": 0.3993553245745406,
                    "add_linear_kernel": False,
                    "power_normalization": False,
                    "hebo_warping": False,
                    "input_warping_type": "kumar",
                    "unused_feature_likelihood": 0.6,
                    "observation_noise": True,
                },
            },
            "epochs": 4,
            "lr": 0.00005,
            "seq_len": 60,
            "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(50),
            "aggregate_k_gradients": 2,
            "nhid": 128,  # orign: 1024
            "steps_per_epoch": 16,
            "weight_decay": 0.0,
            "train_mixed_precision": False,
            "efficient_eval_masking": True,
            "criterion": bar_distribution.FullSupportBarDistribution(
                torch.linspace(-6, 6, 1000)
            ),
            "nlayers": 12,
        }

        r = train(**c)
