from pfn_env import ArtificialEnv
from stable_baselines3 import PPO
import gymnasium as gym
import argparse
import numpy as np
import torch
import random


def test_policy(policy, seed):
    test_episode_num = 10
    env = gym.make("CartPole-v1")
    r_list = []
    for i in range(test_episode_num):
        s = seed + (i * 10)
        obs, info = env.reset(seed=s)
        done = False
        r_sum = 0.
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            r_sum += reward
        r_list.append(r_sum)
    return sum(r_list) / 10.


def training_callback(one, two):
    # These are the steps taken since beginning of training
    # Later for SE training this call back should test every 1000 or so to get measure of real performance
    if one["self"].__dict__["num_timesteps"] % 2048 == 0 or one["self"].__dict__["num_timesteps"] == 1:
        score = test_policy(one["self"], one["self"].__dict__["seed"])
        print(score)
    return one, two


def train_policy_on_se(env_name, time_steps, seed):
    # Parallel environments
    env = ArtificialEnv(env_name)
    # env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=time_steps, callback=training_callback, progress_bar=True)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training policy on One-Shot World Model')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=10000)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_policy_on_se(env_name=args.env, time_steps=args.timestep, seed=args.seed)

