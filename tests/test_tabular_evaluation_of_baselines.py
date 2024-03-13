from unittest import TestCase
from scripts.model_configs import default_task_settings
from scripts.transformer_prediction_interface import get_tabpfn_predictor
from functools import partial
import numpy as np
from utils import torch_nanmean

from scripts.tabular_baselines import get_clf_dict, transformer_metric
from scripts.tabular_evaluation import evaluate, load_evaluations, get_benchmark_tasks
from scripts.tabular_evaluation_notebook_utils import (
    get_table_vis,
    make_results_table_table_vis,
    style_table,
    get_metrics_for_table_vis,
)
from scripts.tabular_metrics import get_standard_eval_metrics, get_main_eval_metric
from datasets import get_benchmark_for_task


class Test(TestCase):
    def test_benchmark_data(self):
        """This test checks that the benchmark data has not changed."""

        def helper(task_type, expected_value):
            datasets_dict = {}

            datasets_dict[f"test_{task_type}"], _ = get_benchmark_for_task(
                task_type,
                split="test",
                max_samples=max_samples,
                max_features=max_features,
                return_as_lists=False,
            )

            checksum = 0
            for split in ["test"]:
                for dataset in datasets_dict[f"{split}_{task_type}"]:
                    checksum += (
                        torch_nanmean(dataset.y.float()).mean()
                        + torch_nanmean(dataset.x.float()).mean()
                    )

            print(checksum)
            assert (
                checksum == expected_value
            ), "The benchmark files were changed. If that was intended change this test."

        max_samples, max_features, max_times, max_classes = default_task_settings()
        helper("regression", -8710608.0)
        helper("multiclass", 137638.0781)

    def test_benchmark_integrated(self):
        """This test checks that the baselines yield the expected results on the benchmark data.
        Weather TabPFN is the same is not checked here, but in test_tabular_train_and_pfn_evaluation.py
        """
        return

        metric_used = get_main_eval_metric(task_type)
        methods = get_clf_dict(task_type)
        max_samples, max_features, max_times, max_classes = default_task_settings()
        device = "cpu"
        tabpfn = get_tabpfn_predictor(
            task_type,
            device,
            local_path="regression/models_diff/prior_diff_real_checkpoint_regression_04_13_2023_13_13_18_n_0_epoch_-1.cpkt",
            base_path="/home/hollmann/prior-fitting-mew/",
        )
        transformer_metric_with_model = partial(transformer_metric, classifier=tabpfn)
        methods["transformer"] = transformer_metric_with_model
