import os
import unittest
import unittest.mock
import numpy as np

from scripts.model_configs import *
from utils import torch_to_python_float

from datasets import get_benchmark_for_task
from scripts.transformer_prediction_interface import get_tabpfn_predictor
from functools import partial
from scripts.tabular_baselines import transformer_metric
from scripts.tabular_evaluation import evaluate
from scripts.tabular_metrics import get_main_eval_metric


class TestTabularTrain(unittest.TestCase):
    @unittest.mock.patch.dict(os.environ, {"TABPFN_CLUSTER_SETUP": "CHARITE"})
    def test_main(self):
        from scripts.hp_tuning_agent import TrainingAgent
        from local_settings import base_path

        """This test checks that
        (1) the TabPFN training yields exactly the same network after 2 training steps
        (2) the TabPFN evaluation yields exactly the same results as the old TabPFN evaluation
        """

        def helper(task_type, expected_value_training, expected_value_evaluation):
            device = "cpu"
            metric_used = get_main_eval_metric(task_type)

            test_datasets, test_datasets_df = get_benchmark_for_task(
                task_type,
                "debug",
                max_samples=max_samples,
                max_features=max_features,
                max_classes=max_classes,
                return_as_lists=False,
            )
            valid_datasets = test_datasets

            config_sample, model_string = load_config(
                config_type=f"tabular_{task_type}", num_features=max_features
            )
            config_sample["nlayers"] = 1
            config_sample["epochs"] = 1
            config_sample["bar_dist_init_batches"] = 1
            config_sample["dataloader_num_workers"] = 0
            config_sample["dataloader_device"] = "cpu"
            # config_sample_['batch_size'] = 24
            config_sample["num_steps"] = 2
            config_sample["aggregate_k_gradients"] = 1
            config_sample["num_buckets"] = 100
            config_sample["emsize"] = 128
            config_sample["lr"] = 0.002
            config_sample["bptt"] = 30

            # Set torch, numpy and random seeds
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)

            agent = TrainingAgent(
                task_type,
                max_features,
                device,
                test_datasets=test_datasets,
                valid_datasets=valid_datasets,
                base_path=base_path,
                model_path="test.ckpt",
                max_classes=10,
                mode="disabled",
                config_sample=config_sample,
            )
            agent.train(continue_model=False)

            self.assertEqual(
                expected_value_training,
                torch_to_python_float(
                    agent.model[2].transformer_encoder.layers[0].linear1.weight.sum()
                ),
                msg="Something in the training loop has changed, check if old training performance "
                "is reached and if yes, update the test.",
            )

            device = "cpu"
            tabpfn = get_tabpfn_predictor(
                task_type,
                device,
                local_path="test.ckpt",
                base_path=base_path,
            )
            transformer_metric_with_model = partial(
                transformer_metric, classifier=tabpfn
            )

            r = evaluate(
                model=transformer_metric_with_model,
                method="transformer",
                task_type=task_type,
                datasets=test_datasets,
                max_time=max_times[0],
                metric_used=metric_used,
                split_number=0,
                save=False,
                bptt=max_samples,
            )
            checksum = 0
            for k, v in r.items():
                checksum += v.pred.mean()
            self.assertEqual(
                expected_value_evaluation,
                checksum,
                "The evaluation has changed. Check if the old evaluation performance is reached "
                "and if yes, update the test.",
            )

        max_samples, max_features, max_times, max_classes = default_task_settings()

        helper(
            "regression",
            expected_value_training=-4.816584587097168,
            expected_value_evaluation=710.6480102539062,
        )
        helper(
            "multiclass",
            expected_value_training=3.6393020153045654,
            expected_value_evaluation=0.3333333432674408,
        )


if __name__ == "__main__":
    unittest.main()
