from pfn_env import ArtificialEnv
from stable_baselines3 import PPO, DQN
import gymnasium as gym
import argparse
import numpy as np
import torch
import random
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.logger import configure
import simple_env


def generate_log_dir_path(env_name, seed):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Make directory for env if not existing
    dir_path = os.path.join(dir_path, "log")
    env_path = os.path.join(dir_path, env_name)
    seed_path = os.path.join(env_path, "seed_" + str(seed))
    return seed_path


def train_policy_on_se(env_name, time_steps, seed):
    # Parallel environments
    env = ArtificialEnv(env_name)
    # env = gym.make(env_name)
    path = generate_log_dir_path(env_name, seed)
    monitor_env = Monitor(env, filename=path)


    model = PPO("MlpPolicy", monitor_env, verbose=0, seed=seed, n_steps=128, stats_window_size=5)

    # Separate evaluation env
    if env_name == "SimpleEnv":
        eval_env = simple_env.SimpleEnv()
    else:
        eval_env = gym.make(env_name)
    monitor_eval_env = Monitor(eval_env, allow_early_resets=True)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(monitor_eval_env, best_model_save_path=path,
                                 log_path=path, eval_freq=100,
                                 deterministic=True, render=False, n_eval_episodes=10)

    new_logger = configure(path, ["json"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=time_steps, callback=eval_callback, progress_bar=True)
    print(monitor_env.get_episode_rewards())
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training policy on One-Shot World Model')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=50000)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_policy_on_se(env_name=args.env, time_steps=args.timestep, seed=args.seed)

