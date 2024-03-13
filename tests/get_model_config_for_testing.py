import os
import random

import numpy as np
import torch.cuda

import utils
from scripts.model_configs import *

base_path = os.path.join("/tmp", "load_model_test")
os.makedirs(os.path.join(base_path, "models_diff"), exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
max_features = 100


def get_prior_config_causal():
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {
        **config_flexible_categorical,
        "num_categorical_features_sampler_a": -1.0,
    }

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config()

    config = {
        **config_general_real_world,
        **config_flexible_categorical_real_world,
        **config_diff,
        **config_gp,
        **config_mlp,
    }

    return config


def get_prior_config_causal_only():
    config = get_prior_config_causal()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 1000000.0,
        "max": 1000001.0,
    }  # Always select MLP
    return config


def get_prior_config_scm_only():
    config = get_prior_config_causal_only()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 1000000.0,
        "max": 1000001.0,
    }  # Always select MLP
    del config["differentiable_hyperparameters"]["is_causal"]
    config["is_causal"] = True
    return config


def get_prior_config_mlp_only():
    config = get_prior_config_causal_only()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 1000000.0,
        "max": 1000001.0,
    }  # Always select MLP
    del config["differentiable_hyperparameters"]["is_causal"]
    config["is_causal"] = False
    return config


def get_prior_config_gp():
    config = get_prior_config_causal()
    config["differentiable_hyperparameters"]["prior_bag_exp_weights_1"] = {
        "distribution": "uniform",
        "min": 0.0,
        "max": 0.0000001,
    }  # Never select MLP
    return config


def get_prior_config(config_type):
    if config_type == "causal":
        return get_prior_config_causal()
    elif config_type == "gp":
        return get_prior_config_gp()
    elif config_type == "bnn":
        raise NotImplemented()
    elif config_type == "causal_only":
        return get_prior_config_causal_only()
    elif config_type == "scm_only":
        return get_prior_config_scm_only()
    elif config_type == "mlp_only":
        return get_prior_config_mlp_only()
    elif config_type == "bag_gp_bnn":
        return get_prior_config_bag_gp_bnn()


def reload_config(
    config_type="causal",
    task_type="multiclass",
    longer=1,
    sample_differentiable_hyperparams_upfront=False,
):
    assert longer
    config = get_prior_config(config_type=config_type)

    config["batch_size"] = CSH.CategoricalHyperparameter(
        "batch_size", [2**i for i in range(6, 9)]
    )

    config["prior_type"], config["differentiable"], config["flexible"] = (
        "prior_bag",
        True,
        True,
    )

    model_string = ""

    if longer == 1:
        config["seq_len"] = CSH.CategoricalHyperparameter("seq_len", [1024, 2048])
        # config['seq_len'] = hp.choice('seq_len', list(range(700, 1200)))
        config["batch_size"] = CSH.CategoricalHyperparameter(
            "batch_size", [2**i for i in range(3, 4)]
        )
        config["emsize"] = CSH.CategoricalHyperparameter(
            "emsize", [2**i for i in range(8, 9)]
        )  ## upper bound is -1

        # Estimate how many gradients need to be aggregated
        # Depends on seq_len, nlayers, emsize and must batch_size > 1
        config["aggregate_k_gradients"] = None

        config["epochs"] = 600
        config["recompute_attn"] = True
        config["bptt_extra_samples"] = None  # changed
        config["dynamic_batch_size"] = False

        model_string = model_string + "_longer"
    elif longer == "hpo":
        config["seq_len"] = CSH.CategoricalHyperparameter("seq_len", [128])  # TODO 1024
        config["emsize"] = 16  # TODO remove
        # config['seq_len'] = hp.choice('seq_len', list(range(700, 1200)))
        config["batch_size"] = CSH.CategoricalHyperparameter(
            "batch_size", [2**i for i in range(3, 4)]
        )

        # Estimate how many gradients need to be aggregated
        # Depends on seq_len, nlayers, emsize and must batch_size > 1
        config["aggregate_k_gradients"] = None

        config["epochs"] = 100  # TODO plz change
        config["recompute_attn"] = True
        config["bptt_extra_samples"] = None  # changed
        config["dynamic_batch_size"] = False

        model_string += "_hpo"

    if sample_differentiable_hyperparams_upfront:
        model_string += "_nondiff"

    if task_type == "multiclass":
        config["max_num_classes"] = 10
        config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])

        config["balanced"] = False
        config["multiclass_loss_type"] = CSH.CategoricalHyperparameter(
            "multiclass_loss_type", ["compatible"]
        )
        model_string += "_multiclass"

    model_string = (
        model_string
        + "_"
        + config_type
        + "_"
        + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        + "_sams"
    )

    return config, model_string


def fix_all_seeds(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_test_config():
    fix_all_seeds()

    config, model_string = reload_config(
        longer=1,
        task_type="multiclass",
        sample_differentiable_hyperparams_upfront=False,
        config_type="causal_only",
    )
    config["lr"] = 0.001

    config["output_multiclass_ordered_p"] = 0.0
    del config["differentiable_hyperparameters"]["output_multiclass_ordered_p"]

    config["multiclass_loss_type"] = "nono"
    config["normalize_to_ranking"] = False  # False

    config["categorical_feature_p"] = 0.2  # diff: .0

    config["nan_prob_a_reason"] = 0.0
    config["nan_prob_no_reason"] = 0.0
    config["nan_prob_unknown_reason"] = 0.0  # diff: .0
    config["set_value_to_nan"] = 0.1  # diff: 1.

    config["normalize_with_sqrt"] = False

    config["new_mlp_per_example"] = True  # actually means false
    config["prior_mlp_scale_weights_sqrt"] = True
    config["batch_size_per_gp_sample"] = None

    config[
        "normalize_ignore_label_too"
    ] = True  # this is weird, but I leave it to be exact

    config["differentiable_hps_as_style"] = False
    config["max_eval_pos"] = 1000

    config["random_feature_rotation"] = True
    config["rotate_normalized_labels"] = True

    config["mix_activations"] = True  # False heisst eig True
    config["nlayers"] = 12
    config["weight_decay"] = 0.0
    config["emsize"] = 512

    # config['emsize'] = 1024
    config["seq_len"] = 1024
    config["canonical_y_encoder"] = False

    opt_gpu_batch_size = 8 * 8
    # config['batch_size'] = opt_gpu_batch_size
    config["num_steps"] = 128
    config["total_available_time_in_s"] = None  # 60*60*22 # 22 hours for some safety...

    config["train_mixed_precision"] = True
    config["efficient_eval_masking"] = True

    config_samples = [evaluate_hypers(config) for _ in range(100)]
    for cs in config_samples:
        for cs2 in config_samples:
            if cs != cs2:
                print([k for k in cs if cs[k] != cs2[k]])
                print(
                    [
                        (
                            k,
                            cs["differentiable_hyperparameters"][k],
                            cs2["differentiable_hyperparameters"][k],
                        )
                        for k in cs["differentiable_hyperparameters"]
                        if cs["differentiable_hyperparameters"][k]
                        != cs2["differentiable_hyperparameters"][k]
                    ]
                )
                raise ValueError("All CSH hyper-paramters should be fixed above.")

    num_gpus_per_node = 1
    jobs = {}
    key = f"TEST_{num_gpus_per_node}x_jobs_fix_{model_string}"
    if config["max_eval_pos"] != 1000:
        key += f"_mep{config['max_eval_pos']}"
    if config["differentiable_hps_as_style"]:
        key += "_addhpstoconf"
    # if not less_diffable:
    # key += '_more_diffable'
    # if config['train_mixed_precision']:
    # key += '_fp16'
    # if config['new_mlp_per_example']:
    # key += '_mlpperex'
    if not config["efficient_eval_masking"]:
        key += "_noem"
    # if config['prior_mlp_scale_weights_sqrt']:
    # key += '_sqrtscalemlp'
    if config["batch_size_per_gp_sample"] is not None:
        key += f"_subbs{config['batch_size_per_gp_sample']}"
    if config["normalize_ignore_label_too"]:
        key += "_normignore"
    if not config["random_feature_rotation"]:
        key += "_nofeatrot"
    if not config["rotate_normalized_labels"]:
        key += "_no_labelrot"
    # if 'output_multiclass_ordered_p' in config:
    #    key += f"_{config['output_multiclass_ordered_p']}muliorderp"
    if config["mix_activations"]:
        key += f"_actnomix"

    if config["seq_len"] != 1024:
        key += f"_{config['seq_len']}bptt"

    addition_configs = {
        "lr": [0.05],
        "emsize": [128],
        "batch_size": [8],
        "normalize_on_train_only": [False, True],
        "normalize_ignore_label_too": [False, True],
        "nlayers": [2],
        "nhead": [2],
        "num_steps": [10],
        "epochs": [3],
        # 'try': list(range(5)),
        # 'warmup_epochs': [80],
        # 'weight_decay': [0.01],
    }

    job_configs = []
    key_additions = []
    for extra_config in utils.product_dict(addition_configs):
        key_addition = "".join(
            f"_{config_name}{config_value}"
            for config_name, config_value in extra_config.items()
        )
        extra_config.pop("try", None)
        config = {**config, **extra_config}

        config["aggregate_k_gradients"] = 4  # 8 is enough for 512 emsize
        assert (128 * 128) % config["batch_size"] == 0
        config_sample = evaluate_hypers(config)
        job_configs.append(
            {"config_sample": config_sample, "name": key + "_n" + key_addition}
        )
        key_additions.append(key_addition)

        # jobs[key_addition] = ex_parallel.submit(train_function, config_sample, key + '_n' + key_addition)

    conf = job_configs[0]

    # only return one config

    return conf
