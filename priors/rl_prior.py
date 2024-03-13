import grid_world
import simple_env
from .prior import Batch
from utils import default_device
import networkx as nx
import random
import numpy as np
import math
import statistics
import gym
import torch
from gym import spaces

def sigmoid(num):
    return 1/(1 + np.exp(-num))


def identity(num):
    return num


def euc_distance(vec):
    summation = 0
    for val in vec:
        summation += val**2
    return math.sqrt(summation)


def get_random_ops(n):
    return list(np.random.choice([np.cos, np.sin, identity, np.tanh], n))


def get_aggregation(n):
    return list(np.random.choice([sum, math.prod, euc_distance, statistics.mean], n))


def get_random_graph(input_num):
    LayerNumbers = [input_num, 4, 1]  # 3 inputs 3 intermediate 1 output

    G = nx.DiGraph()
    total_num_nodes = 0
    e = []
    for i, l in enumerate(LayerNumbers):
        G.add_nodes_from(np.arange(l) + total_num_nodes)
        total_num_nodes += l
        if i < len(LayerNumbers) - 1:
            for k in range(l):
                if i < len(LayerNumbers) - 2:
                    low = 1
                else:
                    low = 0
                #print("Low draw", low, "high_draw", LayerNumbers[i+1])
                number_of_draws = np.random.randint(low, LayerNumbers[i+1] + 1)
                #print("number_of_draws: ", number_of_draws)
                pool_of_target_nodes = np.arange(LayerNumbers[i+1])+total_num_nodes
                #print("pool_of_target_nodes: ", pool_of_target_nodes)
                target_nodes = np.random.choice(pool_of_target_nodes,
                                                number_of_draws,
                                                replace=False)
                #print("drawn node: ", target_nodes)
                for t in target_nodes:
                    #print((total_num_nodes - l + k, t))
                    e.append((total_num_nodes - l + k, t))

    G.add_edges_from(e)
    nx.set_node_attributes(G, 1., "value")

    agg = get_aggregation(len(G.nodes))
    nx.set_node_attributes(G, dict(zip(G.nodes, agg)), name="aggregation")

    ops = get_random_ops(len(G.edges))
    nx.set_edge_attributes(G, dict(zip(G.edges, ops)), name="operation")
    return G


def set_state_and_action(G, s, a):
    s.append(a)
    nx.set_node_attributes(G, dict(zip(np.arange(len(s)), s)), "value")
    return G


def evaluate_graph(RG, s, a):
    RG = set_state_and_action(RG, s, a)
    for n in RG:
        in_edges = RG.in_edges(n)
        if in_edges:
            parent_values = []
            for e in in_edges:
                parent_values.append(RG.edges[e]["operation"](RG.nodes[n]["value"]))
            RG.nodes[n]["value"] = RG.nodes[n]["aggregation"](parent_values)
    return RG.nodes[len(RG.nodes)-1]["value"]


class RandomEnv(gym.Env):

    def __init__(self):
        self.state = None

        self.constant_reward = random.random() > 0.5
        if random.random() > 0.5:
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(
                low=-1., high=1, shape=(1,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 6)

        self.graph_list = []
        for i in range(self.obs_size + 1):
            self.graph_list.append(get_random_graph(self.obs_size + 1))

        self.eps_steps = 0

    def step(self, action):
        next_state_and_reward = []
        for g in self.graph_list:
            next_state_and_reward.append(evaluate_graph(g, list(self.state), action))
        self.state = next_state_and_reward[:self.obs_size]
        self.eps_steps += 1
        if self.constant_reward:
            reward = 1.
        else:
            reward = next_state_and_reward[-1]
        terminated = self.eps_steps > 50
        return np.array(self.state), reward, terminated, False, None

    def render(self):
        pass

    def reset(self, **kwargs):
        self.state = np.random.rand(self.obs_size)
        self.eps_steps = 0
        return self.state, None


class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return

    def forward(self, x):
        return torch.sin(x)


def get_random_activation():
    act_fun = np.random.choice([torch.nn.ReLU(), SinActivation(), torch.nn.Tanh(), torch.nn.Sigmoid()])
    return act_fun


def NNgenerator(input_size, target_num=1):
    model = torch.nn.Sequential(torch.nn.Linear(input_size, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, target_num))


    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            # torch.nn.init.xavier_normal(m.weight, gain=1.5)
            torch.nn.init.uniform(m.weight, -5, 5)
            # m.bias.data.fill_(0.01)

    # model.apply(init_weights)

    return model.float()


class NNEnvironment(gym.Env):

    def __init__(self):
        self.state = None

        self.constant_reward = random.random() > 0.5
        self.discrete = False
        if random.random() > 0.5:
            self.action_dim = 1
            self.discrete = True
            dim = np.random.randint(2, high=5)
            self.action_space = spaces.Discrete(dim)
            self.offset = np.random.choice([0, 0.5, 1.])
            self.scale = np.random.randint(1, 5)
        else:
            self.action_dim = 1 # np.random.randint(1, high=4)
            max_action = np.random.randint(1, high=5)
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(self.action_dim,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 6)
        self.state_scale = 10 * np.random.rand(self.obs_size)
        self.state_offset = 1.5 * (np.random.rand() - 0.5)
        self.total_steps = 0


        self.NN_list = []
        for i in range(self.obs_size):
            self.NN_list.append(NNgenerator(self.obs_size + self.action_dim, target_num=1))
        self.reward_model = NNgenerator(self.obs_size + self.action_dim, target_num=1)
        self.eps_steps = 0

    def step(self, action):
        if self.discrete:
            action = (action - self.offset) * self.scale
        next_state_and_reward = []
        if isinstance(action, int):
            action = [action]
        else:
            action = list(action)
        state_action = torch.tensor(list(self.state) + action).float()
        with torch.no_grad():
            for g in self.NN_list:
                next_state_and_reward.append(g.forward(state_action).item())
            next_state_and_reward.append(self.reward_model(torch.tensor(next_state_and_reward + action).float()))
        self.state = next_state_and_reward[:self.obs_size]
        self.eps_steps += 1
        if self.constant_reward:
            reward = 1.
        else:
            reward = next_state_and_reward[-1]
        if self.total_steps > 1000:
            term_steps = 100
        else:
            term_steps = 50
        terminated = self.eps_steps > term_steps
        self.total_steps += 1
        return np.array(self.state), reward, terminated, False, None

    def render(self):
        pass

    def reset(self, shift=False, **kwargs):
        if shift:
            m = (.5 * np.random.rand()) + 1
            n = (.5 * np.random.rand()) + 1
        else:
            m, n = 1., 1.
        self.state = (np.random.rand(self.obs_size) - m * self.state_offset) * n * self.state_scale
        self.eps_steps = 0
        return self.state, None



def get_dataset(test=False):
    if test:
        env_name = "CartPole-v1"
        # env_name = "Acrobot-v1"
        # env_name = "Pendulum-v1"
        # env_name = 'MountainCarContinuous-v0'
        # env_name = 'MountainCar-v0'
        # env_name = "GridWorld"
        # env_name = "SimpleEnv"
        # env_name = "Reacher-v4"
    else:
        # env_name = np.random.choice(["RandomEnv", "Acrobot-v1", "Pendulum-v1", 'MountainCarContinuous-v0', 'MountainCar-v0'])
        # "Acrobot-v1",
        # env_name = "CartPole-v0"
        env_name = "NNEnv"
    if env_name == "RandomEnv":
        env = RandomEnv()
    elif env_name == "NNEnv":
        env = NNEnvironment()
    elif env_name == "GridWorld":
        env = grid_world.GridWorld()
    elif env_name == "SimpleEnv":
        env = simple_env.SimpleEnv()
    else:
        env = gym.make(env_name)
    if env_name == "CartPole-v0":
        if test:
            print("WARNING - Sampling Cartpole - With no change in the HPs!!!!")
        else:
            env.gravity = np.random.rand() * 20. + 5.
            env.length = np.random.rand() + 2. + 0.5
            env.masspole = np.random.rand() + 0.1 + 0.05
            env.masscart = np.random.rand() * 1. + 0.5
    if env_name == "Pendulum-v1":
        if not test:
            env.g = np.random.rand() * 20. + 5.
            env.m = np.random.rand() * 1. + .5
            env.l = np.random.rand() * 1. + .5
    if env_name == 'MountainCarContinuous-v0':
        if not test:
            env.power = np.random.rand() * 0.001 + 0.001
    if env_name == 'MountainCar-v0':
        if not test:
            env.force = np.random.rand() * 0.001 + 0.0005
            env.gravity = np.random.rand() * 0.0025 + 0.00125
    if env_name == "Acrobot-v1":
        if not test:
            env.LINK_LENGTH_1 = np.random.rand() * 2. + .5
            env.LINK_LENGTH_2 = np.random.rand() * 2. + .5
            env.LINK_MASS_1 = np.random.rand() * 2. + .5
            env.LINK_MASS_1 = np.random.rand() * 2. + .5
    return env


@torch.no_grad()
def get_batch(
        batch_size,
        seq_len,
        num_features,
        device=default_device,
        hyperparameters=None,
        **kwargs
):
    X = torch.full((seq_len, batch_size, num_features), 0.)
    Y = torch.full((seq_len, batch_size, num_features), float(-100.))

    if hyperparameters["test"]:
        X, Y, x_means, x_stds, y_means, y_stds = get_test_batch(seq_len, batch_size, num_features, X, Y)
    else:
        X, Y = get_train_batch(seq_len, batch_size, num_features, X, Y)

    perm = torch.randperm(1000)
    X[:1000, :, :] = X[:1000, :, :][perm]
    Y[:1000, :, :] = Y[:1000, :, :][perm]

    perm = torch.randperm(500)
    X[1000:, :, :] = X[1000:, :, :][perm]
    Y[1000:, :, :] = Y[1000:, :, :][perm]

    if not hyperparameters["test"]:
        return Batch(x=X, y=Y, target_y=Y)
    else:
        return Batch(x=X, y=Y, target_y=Y), x_means, x_stds, y_means, y_stds


def get_test_batch(seq_len, batches, num_features, X, Y):
    for b in range(batches):
        env = get_dataset(True)
        observation, info = env.reset()
        steps_after_done = 0
        ep = 0
        for i in range(seq_len):
            #if ep <= 10:
            #    action = 0
            #elif ep <= 20:
            #    action = 1
            #elif ep <= 30:
            #    act = 2
            #elif ep <= 40:
            #    action = 3
            #else:
            action = env.action_space.sample()

            if isinstance(action, int):
                action = np.array([action])  # TODO detect action type

            act = torch.full((1,), 0.)
            act[:action.shape[0]] = torch.tensor(action)

            obs = torch.full((num_features - 1,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)
            obs_action_pair = torch.hstack((obs, act))
            batch_features = observation.shape[0] + 1  # action.shape[0]
            X[i, b] = obs_action_pair  # * num_features/batch_features TODO compare performance woth or Without
            observation, reward, terminated, truncated, info = env.step(action.item())

            re = torch.full((1,), 0.)
            re[0:] = torch.tensor(reward)  # Reward always 1-D signal num features always same size as in input

            obs = torch.full((num_features - 1,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)
            next_state_reward_pair = torch.hstack((obs, torch.tensor(reward)))
            # next_state_reward_pair = torch.hstack((obs, re)) if reacher size
            Y[i, b] = next_state_reward_pair
            if terminated or truncated:
                steps_after_done += 1
                if steps_after_done >= 0:
                    steps_after_done = 0
                    observation, info = env.reset()
                    ep += 1

        env.close()

    X[1000:, :, :] = X[1000:, 0, :].unsqueeze(1)
    Y[1000:, :, :] = Y[1000:, 0, :].unsqueeze(1)
    x_means = torch.mean(X[:1000, :, :], dim=0)
    x_stds = torch.std(X[:1000, :, :], dim=0)
    X = torch.nan_to_num((X - x_means) / x_stds, nan=0)

    y_means = torch.mean(Y[:1000, :, :], dim=0)
    y_stds = torch.std(Y[:1000, :, :], dim=0)
    Y = torch.nan_to_num((Y - y_means) / y_stds, nan=0)

    return X, Y, x_means, x_stds, y_means, y_stds


def get_train_batch(seq_len, batches, num_features, X, Y):
    for b in range(batches):
        env = get_dataset(False)
        feature_order = torch.randperm(num_features-1)
        action_order = torch.randperm(3)
        observation, info = env.reset()
        for i in range(seq_len):
            action = env.action_space.sample()
            if isinstance(action, int):
                action = [action]
            else:
                action = list(action)
            action_length = len(action)
            act = torch.tensor(action) #  + (1 - len(action)) * [0.])
            # Get X value which is state and action
            # observation is state and filled with -100.
            # Then shuffled
            # Action alsways at the end of the pair
            obs = torch.full((num_features-1,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)
            obs_action_pair = torch.hstack((obs[feature_order], act))  #[action_order]))
            batch_features = observation.shape[0] + action_length
            X[i, b] = obs_action_pair  # * num_features/batch_features TODO compare performance woth or Without

            # Same logic for Y with next action and reward
            # observation is next state and filled with -100.
            # Then shuffled
            # Reward alsways at the end of the pair
            observation, reward, terminated, truncated, info = env.step(action)

            obs = torch.full((num_features - 1,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)

            re = torch.full((1,), 0.)
            re[0:] = torch.tensor(reward)  # Reward always 1-D signal num features always same size as in input

            next_state_reward_pair = torch.hstack((obs[feature_order], re))
            Y[i, b] = next_state_reward_pair
            if terminated:
                observation, info = env.reset(shift=(i > 1000))

        env.close()
        # add gaussian noise
        mean = torch.mean(X[:1000, b], dim=0)
        std = torch.std(X[:1000, b], dim=0)
        X[:, b] = torch.nan_to_num((X[:, b] - mean) / std, nan=0)
        X = X  # + torch.normal(mean=0, std=0.01, size=X.shape)
        mean = torch.mean(Y[:1000, b], dim=0)
        std = torch.std(Y[:1000, b], dim=0)
        Y[:, b] = torch.nan_to_num((Y[:, b] - mean) / std, nan=0)
        Y = Y  # + torch.normal(mean=0, std=0.01, size=Y.shape)
    return X, Y
