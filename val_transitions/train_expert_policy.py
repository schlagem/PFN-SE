import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import argparse
import sys

sys.path.append('..')

from simple_env import SimpleEnv


# TODO summarize this function somewhere (2/3)
def get_env(env_name):
    if env_name == "SimpleEnv":
        return SimpleEnv()
    else:
        env = gym.make(env_name)
    return env


def training_callback(one, two):
    # These are the steps taken since beginning of training
    # Later for SE training this call back should test every 1000 or so to get measure of real performance
    print(one["self"].__dict__["num_timesteps"])
    return one, two


def train_expert_policy(env_name, time_steps):
    # Parallel environments
    env = get_env(env_name)
    print(env.__dict__)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=time_steps)
    model.save("expert_policies/PPO_" + env_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train expert policies')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--t", type=int, default=100000)
    args = parser.parse_args()
    train_expert_policy(args.env, args.t)