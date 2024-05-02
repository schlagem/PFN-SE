import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
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

    # Separate evaluation env
    if env_name == "SimpleEnv":
        eval_env = SimpleEnv()
    else:
        eval_env = gym.make(env_name)
    monitor_eval_env = Monitor(eval_env, allow_early_resets=True)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(monitor_eval_env, eval_freq=1000,
                                 deterministic=True, render=False, n_eval_episodes=10)

    model.learn(total_timesteps=time_steps, callback=eval_callback)
    model.save("expert_policies/PPO_" + env_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train expert policies')
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--t", type=int, default=50000)
    args = parser.parse_args()
    train_expert_policy(args.env, args.t)