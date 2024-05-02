import grid_world
from .prior import Batch
from utils import default_device
import random
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return

    def forward(self, x):
        return torch.sin(x)


class NoOpActivation(torch.nn.Module):
    def __init__(self):
        super(NoOpActivation, self).__init__()
        return

    def forward(self, x):
        return x


def get_random_activation(relu=True, sin=True, tanh=True, sigmoid=True):
    act_choices = []
    if relu:
        act_choices.append(torch.nn.ReLU())
    if sin:
        act_choices.append(SinActivation())
    if tanh:
        act_choices.append(torch.nn.Tanh())
    if sigmoid:
        act_choices.append(torch.nn.Sigmoid())
    if not relu and not sin and not tanh and not sigmoid:
        return NoOpActivation()
    act_fun = np.random.choice(act_choices)
    return act_fun


def NNgenerator(input_size, target_num=1):
    model = torch.nn.Sequential(torch.nn.Linear(input_size, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, 32),
                                get_random_activation(),
                                torch.nn.Linear(32, target_num))

    return model.float()


class CustomFixedDropout(torch.nn.Module):

    def __init__(self, size, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # https://discuss.pytorch.org/t/how-to-fix-the-dropout-mask-for-different-batch/7119/2
        # generate a mask in shape hidden
        self.mask = torch.bernoulli(torch.full((size,), p))

    def forward(self, x):
        return x * self.mask


class HPStepNN(torch.nn.Module):

    def __init__(self, input_size, output_size, hps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_hidden = hps["num_hidden"]
        width_hidden = hps["width_hidden"]
        use_bias = hps["use_bias"]

        self.residual_flag = hps["use_res_connection"]

        self.in_lin = torch.nn.Linear(input_size, width_hidden, bias=use_bias)
        self.in_act = get_random_activation(relu=hps["relu"],
                                            sin=hps["sin"],
                                            tanh=hps["tanh"],
                                            sigmoid=hps["sigmoid"]
                                            )
        self.layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["dropout_p"]))
            if hps["use_layer_norm"]:
                seq_list.append(torch.nn.LayerNorm(width_hidden))
            seq_list.append(get_random_activation(relu=hps["relu"],
                                                  sin=hps["sin"],
                                                  tanh=hps["tanh"],
                                                  sigmoid=hps["sigmoid"]
                                                  ))
            self.layer_list.append(torch.nn.Sequential(*seq_list))
        self.out_lin = torch.nn.Linear(width_hidden, output_size, bias=use_bias)

    def forward(self, x):
        residual = self.in_lin(x)  # TODO norm, mask, dropout
        out = self.in_act(residual)
        for layer in self.layer_list:
            out = layer(out) + self.residual_flag * residual
        return self.out_lin(out)


def SmallNNGen(input_size, output_size):
    model = torch.nn.Sequential(torch.nn.Linear(input_size, 64),
                                get_random_activation(),
                                torch.nn.Linear(64, output_size))

    return model.float()


class FullNNEnv(gym.Env):

    def __init__(self, hps):
        self.state = None

        self.discrete = False
        if random.random() > 0.85:
            self.action_dim = 1
            self.discrete = True
            dim = np.random.randint(2, high=5)
            self.action_space = spaces.Discrete(dim)
        else:
            self.action_dim = np.random.randint(1, high=4)
            max_action = np.random.randint(1, high=5, size=self.action_dim)
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(self.action_dim,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 11)

        self.state_scale = hps["state_scale"] * np.random.rand(self.obs_size)
        self.state_offset = hps["state_offset"] * (np.random.rand() - 0.5)

        self.total_steps = 0

        self.state_to_hidden = HPStepNN(self.obs_size, 64, hps)
        self.action_to_hidden = HPStepNN(self.action_dim, 64, hps)
        self.hidden_to_nextstate = HPStepNN(64, self.obs_size, hps)
        self.reward_model = HPStepNN(2 * self.obs_size + self.action_dim, 1, hps)
        self.eps_steps = 0

    def step(self, action):
        if isinstance(action, int):
            action = [action]
        else:
            action = list(action)
        with torch.no_grad():
            hidden_action = self.action_to_hidden(torch.tensor(action).float())
            hidden_state = self.state_to_hidden(torch.tensor(self.state).float())
            new_state = self.hidden_to_nextstate(hidden_state + hidden_action)
            reward = self.reward_model.forward(torch.cat((new_state, torch.tensor(self.state).float(), torch.tensor(action).float())))
        self.state = new_state.numpy()
        terminated = False
        return self.state, reward, terminated, False, None

    def render(self):
        pass

    def reset(self, **kwargs):
        self.state = (np.random.rand(self.obs_size) - self.state_offset) * self.state_scale
        self.eps_steps = 0
        return self.state, None


class NNEnvironment(gym.Env):

    def __init__(self, hps):
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
            self.action_dim = np.random.randint(1, high=4)
            max_action = np.random.randint(1, high=5)
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(self.action_dim,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 11)
        self.state_scale = hps["state_scale"] * np.random.rand(self.obs_size)
        self.state_offset = hps["state_offset"] * (np.random.rand() - 0.5)
        self.total_steps = 0

        self.NN_list = []
        for i in range(self.obs_size):
            self.NN_list.append(HPStepNN(self.obs_size + self.action_dim, output_size=1, hps=hps))
        self.reward_model = HPStepNN(2 * self.obs_size + self.action_dim, output_size=1, hps=hps)
        self.eps_steps = 0

    def step(self, action):
        if self.discrete:
            action = (action - self.offset) * self.scale
        next_state_and_reward = []
        if isinstance(action, int) or isinstance(action, float):
            action = [action]
        else:
            action = list(action)
        state_action = torch.tensor(list(self.state) + action).float()
        with torch.no_grad():
            for g in self.NN_list:
                next_state_and_reward.append(g.forward(state_action).item())
            next_state_and_reward.append(self.reward_model(torch.tensor(list(self.state) + next_state_and_reward + action).float()))
        self.state = next_state_and_reward[:self.obs_size]
        self.eps_steps += 1
        if self.constant_reward:
            reward = 1.
        else:
            reward = next_state_and_reward[-1]

        term_steps = 50
        terminated = self.eps_steps > term_steps
        self.total_steps += 1
        return np.array(self.state), reward, terminated, False, None

    def render(self):
        pass

    def reset(self, **kwargs):
        self.state = (np.random.rand(self.obs_size) - self.state_offset) * self.state_scale
        self.eps_steps = 0
        return self.state, None



def get_dataset(hps):
    if hps["test"]:
        env_name = "CartPole-v1"
    else:
        env_name = hps["env_name"]
    if env_name == "NNEnv":
        env = NNEnvironment(hps)
    elif env_name == "FullNNEnv":
        env = FullNNEnv(hps)
    else:
        env = gym.make(env_name)
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
    if type(hyperparameters["test"]) == str:
        print("here")
    X = torch.full((seq_len, batch_size, num_features), 0.)
    Y = torch.full((seq_len, batch_size, num_features), float(0.))

    if hyperparameters["test"]:
        X, Y, x_means, x_stds, y_means, y_stds = get_test_batch(seq_len, batch_size, num_features, X, Y,
                                                                hyperparameters)
    else:
        X, Y = get_train_batch(seq_len, batch_size, num_features, X, Y, hyperparameters)

    # TODO get hyperparameters to train len and min trainlen
    perm = torch.randperm(seq_len)
    X = X[perm]
    Y = Y[perm]

    if not hyperparameters["test"]:
        return Batch(x=X, y=Y, target_y=Y)
    else:
        return Batch(x=X, y=Y, target_y=Y), x_means, x_stds, y_means, y_stds


def get_test_batch(seq_len, batches, num_features, X, Y, hps):
    for b in range(batches):
        env = get_dataset(hps)
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


def get_train_batch(seq_len, batches, num_features, X, Y, hps):
    for b in range(batches):
        env = get_dataset(hps)
        feature_order = torch.randperm(num_features-3)
        action_order = torch.randperm(3)
        observation, info = env.reset()
        for i in range(seq_len):
            action = env.action_space.sample()
            if isinstance(action, int) or isinstance(action, np.integer):
                action = [action]
            else:
                action = list(action)
            action_length = len(action)
            act = torch.tensor(+ (3 - len(action)) * [0.] + action)
            # Get X value which is state and action
            # observation is state and filled with -100.
            # Then shuffled
            # Action alsways at the end of the pair
            obs = torch.full((num_features-3,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)
            obs_action_pair = torch.hstack((obs[feature_order], act[action_order]))
            batch_features = observation.shape[0] + action_length
            X[i, b] = obs_action_pair  # * num_features/batch_features TODO compare performance woth or Without

            # Same logic for Y with next action and reward
            # observation is next state and filled with -100.
            # Then shuffled
            # Reward alsways at the end of the pair
            observation, reward, terminated, truncated, info = env.step(action)

            obs = torch.full((num_features - 3,), 0.)
            obs[:observation.shape[0]] = torch.tensor(observation)

            re = torch.full((3,), 0.)
            re[-1] = reward  # Reward always 1-D signal num features always same size as in input

            next_state_reward_pair = torch.hstack((obs[feature_order], re))
            Y[i, b] = next_state_reward_pair
            if terminated:
                observation, info = env.reset()

        env.close()
        if not hps["no_norm"]:
            mean = torch.mean(X[:, b], dim=0)
            std = torch.std(X[:, b], dim=0)
            X[:, b] = torch.nan_to_num((X[:, b] - mean) / std, nan=0)
            mean = torch.mean(Y[:, b], dim=0)
            std = torch.std(Y[:, b], dim=0)
            Y[:, b] = torch.nan_to_num((Y[:, b] - mean) / std, nan=0)
    return X, Y