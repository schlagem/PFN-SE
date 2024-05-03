import math
import time

import gymnasium as gym


import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO

import grid_world
from train import build_model
import encoders
import simple_env

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_done_func(env_name):
    if env_name == "CartPole-v1" or env_name == "CartPole-v0":
        def cartpole_reset(state, steps):
            # Limits for resetting
            theta_threshold_radians = 12 * 2 * math.pi / 360
            x_threshold = 2.4
            x = state[0]
            theta = state[2]
            done = bool(
                x < -x_threshold
                or x > x_threshold
                or theta < -theta_threshold_radians
                or theta > theta_threshold_radians
                or steps > 199
            )
            return done
        return cartpole_reset
    elif env_name == "Pendulum-v1":
        def pendulum_reset(state, steps):
            return steps > 199
        return pendulum_reset
    elif env_name == "SimpleEnv":
        def simple_env_reset(state, steps):
            return steps > 20
        return simple_env_reset
    elif env_name == "Reacher-v4":
        def reacher_reset(state, steps):
            return steps > 50
        return reacher_reset
    elif env_name == "MountainCar-v0":
        def mountain_car_reset(state, steps):
            return state[0] >= 0.5 or steps > 199
        return mountain_car_reset
    else:
        raise NotImplementedError


class ArtificialEnv(gym.Env):

    def __init__(self, env_name, render_mode=None):

        # Create environment to reset and create train samples
        if env_name == "SimpleEnv":
            self.real_env = simple_env.SimpleEnv()
        else:
            self.real_env = gym.make(env_name)

        self.action_space = self.real_env.action_space
        self.observation_space = self.real_env.observation_space

        self.done_func = get_done_func(env_name)
        self.episode_steps = 0

        num_features = 14
        batch_size = 1

        seq_len = 1001
        self.train_x = torch.full((seq_len, batch_size, num_features), 0.)
        self.train_y = torch.full((seq_len, batch_size, num_features), float(0.))
        observation, info = self.real_env.reset()
        for b in range(batch_size):
            ep = 0
            steps_after_done = 0
            for i in range(1001):
                if ep < 7:
                    action = 0
                elif ep < 15:
                    action = 1
                else:
                    action = self.real_env.action_space.sample()
                if isinstance(action, int) or isinstance(action, np.int64):
                    action_array = np.array([action])  # TODO detect action type
                else:
                    action_array = action
                act = torch.full((3,), 0.)
                act[:action_array.shape[0]] = torch.tensor(action_array)

                obs = torch.full((num_features - 3,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                obs_action_pair = torch.hstack((obs, act))

                # obs = torch.full((num_features - 1,), 0.)
                # obs[:observation.shape[0]] = torch.tensor(observation)
                # obs_action_pair = torch.hstack((obs, torch.tensor(action)))

                # batch_features = observation.shape[0] + 1  # action.shape[0]
                self.train_x[i, b] = obs_action_pair # * num_features / batch_features
                observation, reward, terminated, truncated, info = self.real_env.step(action)

                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                # obs[-1] = float(terminated or truncated) # TODO if possible for all env
                next_state_reward_pair = torch.hstack((obs, torch.tensor(reward)))
                self.train_y[i, b] = next_state_reward_pair
                if terminated or truncated or i % 50 == 0:
                    steps_after_done += 1
                    if steps_after_done >= 0:
                        steps_after_done = 0
                        observation, info = self.real_env.reset()
                        ep += 1
                        print(f"Episode {ep}")

        self.train_y = self.train_y.to(device)
        self.train_x = self.train_x.to(device)

        # Normalize the columns to 0 mean and 1 Variance
        self.x_mean = torch.mean(self.train_x[:1000, :], dim=0)
        self.x_std = torch.std(self.train_x[:1000, :], dim=0)
        self.train_x = torch.nan_to_num((self.train_x - self.x_mean) / self.x_std, nan=0)

        self.y_mean = torch.mean(self.train_y[:1000, :], dim=0)
        self.y_std = torch.std(self.train_y[:1000, :], dim=0)
        self.train_y = torch.nan_to_num((self.train_y - self.y_mean) / self.y_std, nan=0)

        # building Transformer model and loading weights
        criterion = nn.MSELoss(reduction='none')
        # TODO test batch?
        hps = {'test': True}
        self.pfn = build_model(
            criterion=criterion,
            encoder_generator=encoders.Linear,
            test_batch=None,
            n_out=14,
            emsize=512, nhead=4, nhid=1024, nlayers=6,
            seq_len=1001,
            y_encoder_generator=encoders.Linear,
            decoder_dict={},
            extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
        ).to(device)
        print(
            f"Using a Transformer with {sum(p.numel() for p in self.pfn.parameters()) / 1000 / 1000:.{2}f} M parameters"
        )
        self.pfn.load_state_dict(torch.load("saved_models/first_incumbent.pt"))
        self.pfn.eval()

        self.state = None

    def step(self, a):
        a = a  # .item()
        # TODO check if all normalizations are correct
        if self.state is None:
            raise gym.error.ResetNeeded

        if isinstance(a, int) or isinstance(a, np.int64):
            action_array = np.array([a])  # TODO detect action type
        else:
            action_array = a
        act = torch.full((3,), 0.)
        act[:action_array.shape[0]] = torch.tensor(action_array)

        obs = torch.full((14 - 3,), 0.)
        obs[:self.state.shape[0]] = torch.tensor(self.state)
        obs = torch.hstack((obs, act)).to(device)

        norm_state_action = torch.nan_to_num((obs - self.x_mean) / self.x_std)
        self.train_x[1000, :, :] = norm_state_action.to(device)
        with torch.no_grad():
            logits = self.pfn(self.train_x[:1000], self.train_y[:1000], self.train_x[:])
            ns = logits[1000, :, :].detach().clone() * self.y_std + self.y_mean
            ns = torch.mean(ns, dim=0)  # TODO discrete steps

        self.state = ns[:self.state.shape[0]].cpu().numpy()
        re = torch.nan_to_num(ns[-1], nan=-1)  # TODO if this accurate enough

        done = self.done_func(self.state, self.episode_steps)
        self.episode_steps += 1
        return self.state, re, done, False, {}

    def reset(self, **kwargs):
        # TODO reset based on real environment
        self.episode_steps = 0
        self.state, information = self.real_env.reset()
        self.train_y[1000:] = 0
        self.train_x[1000:] = 0
        return self.state, information

