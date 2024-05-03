import torch
from priors import rl_prior
import argparse
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env


vec_env = make_vec_env("MountainCar-v0")

model = PPO.load("val_transitions/expert_policies/PPO_MountainCar-v0.zip")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
