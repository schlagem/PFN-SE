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

from priors.prior import Batch
from train import build_model
import encoders
from numpy import cos, pi, sin

import time


class SimpleEnv(gym.Env):

    def __init__(self, render_mode=None):

        self.state = None
        self.episode_steps = 0
        self.action_space = gym.spaces.Box(low=-10, high=10)

    def step(self, a):
        self.state += a
        self.state = self.state.clip(-30, 30)
        self.episode_steps += 1
        re = -1. * np.abs(self.state).item()
        done = self.episode_steps == 20  # or self.state < .1
        return self.state, re, done, False, None

    def reset(self, **kwargs):
        # TODO reset based on real environment
        self.episode_steps = 0
        self.state = 10 * (np.random.rand(1) - .5)
        return self.state, None

    def render(self):
        print(60 * "#")
        print(np.round(self.state))
        world = 29 * ["-"] + ["**"] + 29 * ["-"]
        world[np.round(self.state) + 30] = "@"
        print("".join(world))
        print(60 * "#")
