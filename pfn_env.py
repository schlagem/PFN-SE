import math
from typing import Optional, Union, List

import gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym.core import RenderFrame
import numpy as np

import grid_world
from priors.prior import Batch
from train import build_model
import encoders
from numpy import cos, pi, sin
import simple_env
import time


#env = gym.make("CartPole-v0")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ArtificialEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    LINK_LENGTH_1 = 1.0  # [m]
    LINK_LENGTH_2 = 1.0  # [m]
    LINK_MASS_1 = 1.0  #: [kg] mass of link 1
    LINK_MASS_2 = 1.0  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.0  #: moments of inertia for both links

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    AVAIL_TORQUE = [-1.0, 0.0, +1]

    torque_noise_max = 0.0

    SCREEN_DIM = 500

    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Limits for reseting
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.episode_steps = 0

        # Create environment to reset and create train samples
        self.real_env = gym.make("CartPole-v0")
        # self.action_space = self.real_env.action_space
        # self.real_env = gym.make("Acrobot-v1")
        # self.real_env = gym.make("Pendulum-v1")
        # self.real_env = simple_env.SimpleEnv()
        # self.real_env = grid_world.GridWorld()
        self.action_space = self.real_env.action_space

        num_features = 7
        batch_size = 1


        seq_len = 1001
        self.train_x = torch.full((seq_len, batch_size, num_features), 0.)
        self.train_y = torch.full((seq_len, batch_size, num_features), float(0.))
        observation, info = self.real_env.reset()
        for b in range(batch_size):
            for i in range(1001):
                high = np.array([4.9, 5., 0.45, 5.0])
                low = -high
                random_state = np.random.uniform(low=low, high=high)
                action = self.real_env.action_space.sample()
                self.real_env.env.env.env.state = random_state.copy()
                observation = self.real_env.env.env.env.state  # _get_obs() for Pendulum e.g. state =/= obs
                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                obs_action_pair = torch.hstack((obs, torch.tensor(action)))
                batch_features = observation.shape[0] + 1  # action.shape[0]
                self.train_x[i, b] = obs_action_pair * num_features / batch_features
                observation, reward, terminated, truncated, info = self.real_env.step(action)

                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                # obs[-1] = float(terminated or truncated) # TODO if possible for all env
                next_state_reward_pair = torch.hstack((obs, torch.tensor(reward)))
                self.train_y[i, b] = next_state_reward_pair

        """
            ep = 0
            steps_after_done = 0
            for i in range(1001):
                if ep <= 7:
                    action = 0
                elif ep <= 14:
                    action = 1
                #elif ep <= 30:
                #    action = 2
                #elif ep <= 40:
                #    action = 3
                else:
                    action = self.real_env.action_space.sample()
                #action_array = np.array([action])
                #act = torch.full((3,), 0.)
                #act[:action_array.shape[0]] = torch.tensor(action_array)

                #obs = torch.full((num_features - 3,), 0.)
                #obs[:observation.shape[0]] = torch.tensor(observation)
                #obs_action_pair = torch.hstack((obs, act))

                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                obs_action_pair = torch.hstack((obs, torch.tensor(action)))
                batch_features = observation.shape[0] + 1  # action.shape[0]
                self.train_x[i, b] = obs_action_pair * num_features / batch_features
                observation, reward, terminated, truncated, info = self.real_env.step(action)

                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                # obs[-1] = float(terminated or truncated) # TODO if possible for all env
                next_state_reward_pair = torch.hstack((obs, torch.tensor(reward)))
                self.train_y[i, b] = next_state_reward_pair
                if terminated or truncated or i % 49 == 0:
                    steps_after_done += 1
                    if steps_after_done >= 7:
                        steps_after_done = 0
                        observation, info = self.real_env.reset()
                        ep += 1
                        print(f"Episode {ep}")
        """

        plt.ioff()
        fig, axs = plt.subplots(1, 6)
        fig.tight_layout()
        # Plotting data to see spread
        for i, ax in enumerate(axs.reshape(-1)):
            t_max = torch.max(torch.abs(self.train_x[:1000, :, i]))
            print(f"Dimension {i} - Max Value - {t_max} ")
            # Center line
            ax.set_ylim([-t_max-0.1, t_max+.1])
            ax.set_xlim([-.1, .1])
            ax.scatter(np.zeros(1000 * batch_size), self.train_x[:1000, :, i], s=3, c="tab:orange")
            ax.set_ylabel("target")
            ax.set_ylabel("prediction")
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        even = num_features % 2 == 0
        three_divisable = num_features % 3 == 0
        if three_divisable:
            fig, axs = plt.subplots(3, num_features//3)
        elif even:
            fig, axs = plt.subplots(2, num_features//2)
        else:
            fig, axs = plt.subplots(1, num_features)
        fig.tight_layout()

        for k, ax in enumerate(axs.reshape(-1)):
            ax.hist(self.train_x[:1000, :, k].view(-1), 100, density=True)

        plt.show()
        plt.clf()
        plt.cla()

        plt.ion()

        # Normalize the columns to 0 mean and 1 Variance
        # TODO 500 as variable -> must match training
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
            n_out=7,
            emsize=512, nhead=4, nhid=1024, nlayers=6,
            # emsize=512, nhead=8, nhid=1024, nlayers=12,
            # emsize=800, nhead=8, nhid=1024, nlayers=12,
            # emsize=1024, nhid=2048, nlayers=8, nhead=8,
            # seq_len=1202,
            seq_len=1001,
            y_encoder_generator=encoders.Linear,
            decoder_dict={},
            extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
        )
        print(
            f"Using a Transformer with {sum(p.numel() for p in self.pfn.parameters()) / 1000 / 1000:.{2}f} M parameters"
        )
        # self.pfn.load_state_dict(torch.load("trained_models/semi_working_prioir_small.pt"))
        # self.pfn.load_state_dict(torch.load("trained_models/prior_larger.pt"))
        # self.pfn.load_state_dict(torch.load("trained_models/se_pfn_transfer.pt"))
        # self.pfn.load_state_dict(torch.load("trained_models/se_pfn_transfer_new_to_not_overwrite.pt"))
        # self.pfn.load_state_dict(torch.load("trained_models/prior_retrain_working.pt"))
        self.pfn.load_state_dict(torch.load("trained_models/prior_weighted_eval_pos_1001_seq_default_y_0.pt"))
        # TODO check if eval needed/wanted
        self.pfn.eval()

        self.state = None

    def step(self, a):
        a = a  # .item()
        # TODO check if all normalizations are correct
        if self.state is None:
            raise gym.error.ResetNeeded

        #action_array = np.array([a])
        #act = torch.full((3,), 0.)
        #act[:action_array.shape[0]] = torch.tensor(action_array)

        #obs = torch.full((14 - 3,), 0.)
        #obs[:self.state.shape[0]] = torch.tensor(self.state)
        #obs = torch.hstack((obs, act))

        # normalize state
        obs = torch.zeros(7)
        obs[:self.state.shape[0]] = torch.tensor(self.state)
        obs[-1] = a
        # print("State:", obs)
        obs = obs * 7 / (self.state.shape[0] + 1)  # TODO detect automatically
        norm_state_action = torch.nan_to_num((obs - self.x_mean) / self.x_std)
        self.train_x[1000, :, :] = norm_state_action
        with torch.no_grad():
            logits = self.pfn(self.train_x[:1000], self.train_y[:1000], self.train_x[:])
            ns = logits[1000, :, :].detach().clone() * self.y_std + self.y_mean
            ns = torch.mean(ns, dim=0)  # TODO discrete steps
        # self.state = torch.round(ns[:self.state.shape[0]]).numpy()
        self.state = ns[:self.state.shape[0]].numpy()
        # print("New state. ", self.state)
        re = torch.nan_to_num(ns[-1], nan=-1)  # TODO if this accurate enough
        # print(re)
        # TODO constant reward is nan
        # TODO detect done

        x = self.state[0]
        theta = self.state[2]
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.episode_steps > 199
        )
        # done = self.episode_steps > 199
        """
        done = bool(
            self.state[0] > 0.5
            or self.episode_steps > 199
        )
        """
        #done = bool(self.episode_steps > 20) or self.state[0] <= 0 or self.state[1] <= 0 or\
        #       self.state[0] >= 6 or self.state[1] >= 7
        #print("done: ", done)
        # print("--------")
        self.episode_steps += 1
        return self.state, re, done, False, None

    def reset(self, **kwargs):
        # TODO reset based on real environment
        self.episode_steps = 0
        self.state, information = self.real_env.reset()
        self.train_y[1000:] = 0
        self.train_x[1000:] = 0
        return self.state, information

    def update_context(self, policy):
        print("Updating second half of context...")
        observation, info = self.real_env.reset()
        num_features = 7
        batch_size = 1
        ep = 0
        fraction = 10
        with torch.no_grad():
            for i in range(500 * fraction):
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(state).max(1)[1].view(1)
                # action, _ = policy(state)
                # action = action.view(1)
                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                obs_action_pair = torch.hstack((obs, action))
                batch_features = observation.shape[0] + 1  # action.shape[0]
                if i % fraction == 0:
                    self.train_x[500 + i//fraction, 0] = obs_action_pair * num_features / batch_features
                #observation, reward, terminated, truncated, info = self.real_env.step(action.detach().cpu().numpy())
                observation, reward, terminated, truncated, info = self.real_env.step(action.item())

                obs = torch.full((num_features - 1,), 0.)
                obs[:observation.shape[0]] = torch.tensor(observation)
                next_state_reward_pair = torch.hstack((obs, torch.tensor(reward)))
                if i % fraction == 0:
                    self.train_y[500 + i//fraction, 0] = next_state_reward_pair
                if terminated or truncated:
                    observation, info = self.real_env.reset()
                    print("Episode:", ep)
                    ep += 1