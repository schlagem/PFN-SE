from unittest import TestCase
import torch
from bar_distribution import (
    HalfSupportBarDistributionForSurvival,
    FullSupportBarDistribution,
)


class Test(TestCase):
    def test_survival_bar_dist(self):
        predictions = (
            torch.tensor(
                [
                    [0, 0, 10, 10],
                    [10, 10, 0, 0],
                    [0, 0, 10, 0],
                    [0, 0, 0, 10],
                    [10, 0, 0, 0],
                    [10, 0, 0, 0],
                    [0, 0, 0, 10],
                    [0, 0, 0, 10],
                    [0, 0, 10, 0],
                    [0, 0, 10, 0],
                    [0, 0, 10, 0],
                    [0, 0, 1, 0],
                ]
            )
            .unsqueeze(1)
            .float()
        )
        event_times = (
            torch.tensor([5, 5, 2, 2, -2, -1, 8, 9, 5, 6, 5, 5])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .float()
        )
        event_observed = (
            torch.tensor(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                ]
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        bar_dist = HalfSupportBarDistributionForSurvival(
            borders=torch.arange(0, 10, 2).float()
        )

        predictions.shape, event_observed.shape

        loss, loss_censored, loss_observed = bar_dist(
            predictions, event_times, event_observed
        )

        assert torch.all(loss_censored[event_observed.squeeze(-1)] == 0.0)
        assert torch.all(loss_observed[~event_observed.squeeze(-1)] == 0.0)
        assert torch.all(loss == loss_censored + loss_observed)

        assert loss_censored[0] < loss_censored[1]
        assert loss_censored[2] == loss_censored[3]
        assert loss_censored[4] < loss_censored[5]
        assert loss_censored[6] < loss_censored[7]
        assert loss_censored[8] == loss_censored[9]

        assert loss_observed[8] == loss_observed[9]
        assert loss_observed[10] < loss_observed[11]
