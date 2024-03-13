import os
import unittest

import torch.cuda

from datasets import get_benchmark_for_task
from scripts.model_builder import save_model, get_model
from scripts.model_configs import *
from scripts.transformer_prediction_interface import (
    TabPFNClassifier,
)

from tests.get_model_config_for_testing import get_test_config, fix_all_seeds


class TestTabularTrain(unittest.TestCase):
    def test_main(self):
        """This is currently untested. We need to make this test simpler."""
        # Simplify with new interfaces
        # Train one epoch
        # Save model
        # Load Model
        # Train one epoch

        return
        base_path = os.path.join("tmp", "load_model_test")
        os.makedirs(os.path.join(base_path, "models_diff"), exist_ok=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

        N_epochs_to_save = 50

        conf, name = list(get_test_config().values())

        number_of_epochs = 4
        number_of_epoch_to_train_before_load_and_store = 2

        def save_callback(
            model,
            epoch,
            data_loader=None,
            optimizer_state=None,
            scaler_state=None,
            **kwargs,
        ):
            done_part = epoch / conf["epochs"]
            if not hasattr(model, "last_saved_epoch"):
                model.last_saved_epoch = 0
            if done_part > model.last_saved_epoch / N_epochs_to_save or done_part > 1.0:
                print(f"Saving model.. epoch {model.last_saved_epoch}")
                conf["done_part_in_training"] = done_part
                save_model(
                    model,
                    base_path,
                    os.path.join(
                        "models_diff",
                        f"prior_diff_real_checkpoint{name}_epoch_{model.last_saved_epoch}.cpkt",
                    ),
                    config_sample=conf,
                    optimizer_state=optimizer_state,
                    scaler_state=scaler_state,
                    **kwargs,
                )
                model.last_saved_epoch += 1
                if (
                    model.last_saved_epoch
                    == number_of_epoch_to_train_before_load_and_store + 1
                ):
                    fix_all_seeds(
                        1
                    )  # to synch the saved model with the model that just trains through all the epochs at once

        name = "test1"
        conf["epochs"] = number_of_epochs
        conf["unittest_active"] = True
        conf["unittest_number_of_epochs"] = number_of_epochs
        conf[
            "unittest_number_of_epoch_saved"
        ] = number_of_epoch_to_train_before_load_and_store

        fix_all_seeds(0)

        # train model for number_of_epochs
        get_model(
            config=conf.copy(),
            device=device,
            should_train=True,
            verbose=1,
            epoch_callback=save_callback,
        )

        # train a model for one epoch store it and load it again
        conf["epochs"] = number_of_epoch_to_train_before_load_and_store
        name = "test_n_0"

        del conf["done_part_in_training"]

        fix_all_seeds(0)

        # train new model for number_of_epoch_to_train_before_load_and_store
        get_model(
            config=conf.copy(),
            device=device,
            should_train=True,
            verbose=1,
            epoch_callback=save_callback,
        )

        del conf  # make sure that the config can't be used

        # load the stored model
        (_, _, model_loaded, _), conf, _ = load_model_workflow(
            0,
            number_of_epoch_to_train_before_load_and_store,
            "test",
            base_path,
            device=device,
            restore_training_capabilities=True,
        )

        # train the model for the remaining epochs
        train_existing_model(
            conf,
            device,
            model_loaded,
            save_callback,
            number_of_epochs_to_train=(
                number_of_epochs - number_of_epoch_to_train_before_load_and_store
            ),
            verbose=True,
        )

        task_type = "multiclass"
        max_samples = 1000
        max_features = 100

        test_datasets, test_datasets_df = get_benchmark_for_task(
            task_type, "test", max_samples=max_samples, max_features=max_features
        )

        extended_path = os.path.join(base_path, "models_diff")
        os.rename(
            os.path.join(
                extended_path,
                f"prior_diff_real_checkpointtest1_epoch_{number_of_epochs}.cpkt",
            ),
            os.path.join(
                extended_path, "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
            ),
        )

        classifier_with_normal_training = TabPFNClassifier(
            device=device, base_path=base_path
        )

        os.remove(
            os.path.join(extended_path, "prior_diff_real_checkpoint_n_0_epoch_100.cpkt")
        )
        os.rename(
            os.path.join(
                extended_path,
                f"prior_diff_real_checkpointtest_n_0_epoch_{number_of_epochs}.cpkt",
            ),
            os.path.join(
                extended_path, "prior_diff_real_checkpoint_n_0_epoch_100.cpkt"
            ),
        )

        classifier_with_loaded_model = TabPFNClassifier(
            device=device, base_path=base_path
        )
        os.remove(
            os.path.join(extended_path, "prior_diff_real_checkpoint_n_0_epoch_100.cpkt")
        )

        for i in range(number_of_epochs):
            os.remove(
                os.path.join(
                    extended_path, f"prior_diff_real_checkpointtest_n_0_epoch_{i}.cpkt"
                )
            )
            os.remove(
                os.path.join(
                    extended_path, f"prior_diff_real_checkpointtest1_epoch_{i}.cpkt"
                )
            )

        print("number of test_datasets:", len(test_datasets))
        for dataset in test_datasets:
            xs, ys = dataset[1].clone(), dataset[2].clone()
            eval_position = xs.shape[0] // 2
            train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
            test_xs, test_ys = xs[eval_position:], ys[eval_position:]

            classifier_with_normal_training.fit(train_xs, train_ys)
            prediction_normal = classifier_with_normal_training.predict_proba(test_xs)

            classifier_with_loaded_model.fit(train_xs, train_ys)
            prediction_with_loaded_model = classifier_with_loaded_model.predict_proba(
                test_xs
            )

            self.assertTrue(
                prediction_normal.shape == prediction_with_loaded_model.shape
            )
            number_of_predictions, number_of_classes = prediction_normal.shape

            for number in range(number_of_predictions):
                for class_nr in range(number_of_classes):
                    # checks that every class probability has difference of at most
                    if (
                        prediction_with_loaded_model[number][class_nr]
                        != prediction_normal[number][class_nr]
                    ):
                        print(
                            prediction_with_loaded_model[number][class_nr]
                            - prediction_normal[number][class_nr]
                        )
                    self.assertEqual(
                        prediction_with_loaded_model[number][class_nr],
                        prediction_normal[number][class_nr],
                    )
        os.remove("206705.0.csv")
        # remove the csv file that stores the datasets


if __name__ == "__main__":
    unittest.main()
