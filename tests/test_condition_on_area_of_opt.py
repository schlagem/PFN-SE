import unittest
from functools import partial
import torch
import bar_distribution
import encoders
import utils
import priors.utils, priors.hebo_prior, priors, priors.condition_on_area_of_opt


class TestNonMyopic(unittest.TestCase):
    @torch.no_grad()
    def test_no_errors(self):
        def get_batch_test(batch_size, seq_len, num_features, *args, **kwargs):
            y = torch.rand(seq_len, batch_size)
            y[torch.arange(batch_size)[None]] += 1.0
            return priors.Batch(
                x=torch.rand(batch_size, seq_len, num_features), target_y=y, y=y
            )

        b: priors.Batch = priors.condition_on_area_of_opt.get_batch(
            batch_size=10,
            seq_len=10,
            num_features=2,
            get_batch=get_batch_test,
            single_eval_pos=None,
            epoch=None,
            device="cpu",
        )
