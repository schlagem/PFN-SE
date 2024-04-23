from pfn_env import ArtificialEnv
from stable_baselines3 import PPO
import gymnasium as gym
import argparse
import numpy as np


def test_policy(policy):
    test_episode_num = 10
    env = gym.make("CartPole-v1")
    r_sum = 0.
    for i in range(test_episode_num):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            r_sum += reward
    return r_sum / 10.


def training_callback(one, two):
    # These are the steps taken since beginning of training
    # Later for SE training this call back should test every 1000 or so to get measure of real performance
    if one["self"].__dict__["num_timesteps"] % 2048 == 0:
        score = test_policy(one["self"])
        print(score)
    return one, two


def train_policy_on_se(env_name, time_steps):
    # Parallel environments
    env = ArtificialEnv(env_name)
    #env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=time_steps, callback=training_callback)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Training policy on One-Shot World Model')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--timestep", type=int, default=10000)
    args = parser.parse_args()
    train_policy_on_se(env_name=args.env, time_steps=args.timestep)

