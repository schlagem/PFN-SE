import time

from .prior import Batch
from utils import default_device
import random
import numpy as np
import gymnasium as gym
import torch
from gymnasium import spaces

torch.set_printoptions(sci_mode=False, precision=1)


def init_func_generator(init_std):
    def init_func(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=init_std)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, std=init_std)
    return init_func


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


class Dym:
    def __init__(self):
        self.dt = 0.01


class VelDym(Dym):
    def __init__(self, dym_type, action_dim):
        super().__init__()
        self.controlling_action_dim = np.random.randint(low=0, high=action_dim)
        self.velocity = None
        self.invert = np.random.choice([True, False])
        self.dym_type = dym_type  # choice out of sin cos x y
        if self.dym_type == "sin" or self.dym_type == "cos" or self.dym_type == "rad":
            self.gravity = np.random.choice([True, False])
            self.g = (5. * np.random.rand()) + 5.  # Max gravity 10.  min gravity 5.
            self.dampening = .9 + 0.1 * np.random.rand()
        if self.dym_type == "y":
            self.gravity = True
            self.g = (5. * np.random.rand()) + 5.  # Max gravity 10.  min gravity 5.
        if self.dym_type == "x":
            self.gravity = False

    def update(self, action, pos_dym):
        self.velocity = self.velocity + action[self.controlling_action_dim] * self.dt
        if self.gravity:
            if self.dym_type == "sin" or self.dym_type == "cos" or self.dym_type == "rad":
                # time 5 to keep desired properties with same action space as lin
                self.velocity -= self.g * np.cos(pos_dym.position) * self.dt * 5
                self.velocity *= self.dampening
            elif self.dym_type == "y":
                self.velocity -= self.g * pos_dym.position * self.dt

        # cap maximum momentum for stability
        max_vel = 7.
        if self.velocity > max_vel:
            self.velocity = max_vel

        if self.velocity < -max_vel:
            self.velocity = -max_vel

    def get_velocity(self):
        return self.velocity * (-1 * self.invert)

    def reset(self):
        self.velocity = 6 * (np.random.rand() - 0.5)
        return self.get_velocity()


class PosDym(Dym):
    def __init__(self, dym_type):
        super().__init__()
        self.position = None  # x/y pos or angle in radians
        self.dym_type = dym_type  # choice out of sin, cos, x, y
        if self.dym_type == "rad":
            self.only_positive = np.random.choice([True, False])
        if self.dym_type == "x":
            self.max_pos = 1.0
            self.min_pos = -1.0
            # choice between inelastic and random elasticity
            self.elasticity = np.random.choice([0., 0.3 + 0.7 * np.random.rand()])

        elif self.dym_type == "y":
            self.max_pos = 1.0
            self.min_pos = 0.0
            self.elasticity = 0.3 + 0.7 * np.random.rand()  # TODO find min max values

    def update(self, vel_dym):
        velocity = vel_dym.velocity
        self.position = self.position + velocity * self.dt
        if self.dym_type == "x" or self.dym_type == "y":
            if self.position > self.max_pos:
                self.position = self.max_pos
                vel_dym.velocity = -self.elasticity * velocity

            if self.position < self.min_pos:
                self.position = self.min_pos
                vel_dym.velocity = -self.elasticity * velocity

    def get_position(self):
        if self.dym_type == "x" or self.dym_type == "y":
            return self.position
        if self.dym_type == "cos":
            return np.cos(self.position)
        if self.dym_type == "sin":
            return np.sin(self.position)
        if self.dym_type == "rad":
            return np.cos(np.arctan2(np.sin(self.position), np.cos(self.position))) + self.only_positive * np.pi

    def reset(self):
        if self.dym_type == "x":
            self.position = np.random.rand() * 2. - 1.  # ranges between -1 and 1
            return self.get_position()
        if self.dym_type == "y":
            self.position = np.random.rand()  # ranges between 0 and 1
            return self.get_position()
        if self.dym_type == "sin" or self.dym_type == "cos" or self.dym_type == "rad":
            self.position = np.random.rand() * 2 * np.pi  # ranges between 0 and 2 pi
            return self.get_position()


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
        residual = self.in_lin(x)
        out = self.in_act(residual) + self.residual_flag * residual
        for layer in self.layer_list:
            out = layer(out) + self.residual_flag * residual
        return self.out_lin(out)


class HPStateActionNN(torch.nn.Module):

    def __init__(self, state_size, action_size, hps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_hidden = hps["num_hidden"]
        width_hidden = hps["width_hidden"]
        use_bias = hps["use_bias"]

        self.residual_flag = hps["use_res_connection"]
        func = init_func_generator(hps["init_std"])

        self.state_in = torch.nn.Linear(state_size, width_hidden, bias=use_bias).apply(func)
        self.state_in_act = get_random_activation(relu=hps["relu"],
                                            sin=hps["sin"],
                                            tanh=hps["tanh"],
                                            sigmoid=hps["sigmoid"]
                                            )
        self.state_layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["dropout_p"]))
            seq_list.append(get_random_activation(relu=hps["relu"],
                                                  sin=hps["sin"],
                                                  tanh=hps["tanh"],
                                                  sigmoid=hps["sigmoid"]
                                                  ))
            self.state_layer_list.append(torch.nn.Sequential(*seq_list).apply(func))
        self.state_out_layer = torch.nn.Linear(width_hidden, 1, bias=use_bias).apply(func)

        self.residual_flag = hps["a_use_res_connection"]

        func = init_func_generator(hps["a_init_std"])

        num_hidden = hps["a_num_hidden"]
        width_hidden = hps["a_width_hidden"]
        use_bias = hps["a_use_bias"]

        self.action_in = torch.nn.Linear(action_size, width_hidden, bias=use_bias).apply(func)
        self.action_in_act = get_random_activation(relu=hps["a_relu"],
                                            sin=hps["a_sin"],
                                            tanh=hps["a_tanh"],
                                            sigmoid=hps["a_sigmoid"]
                                            )
        self.action_layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["a_use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["a_dropout_p"]))
            seq_list.append(get_random_activation(relu=hps["a_relu"],
                                                  sin=hps["a_sin"],
                                                  tanh=hps["a_tanh"],
                                                  sigmoid=hps["a_sigmoid"]
                                                  ))
            self.action_layer_list.append(torch.nn.Sequential(*seq_list).apply(func))
        self.action_out_layer = torch.nn.Linear(width_hidden, 1, bias=use_bias).apply(func)

    def forward(self, s, a):
        state_residual = self.state_in(s)
        s_out = self.state_in_act(state_residual) + self.residual_flag * state_residual
        for layer in self.state_layer_list:
            s_out = layer(s_out) + self.residual_flag * state_residual
        state_out = self.state_out_layer(s_out)

        action_residual = self.action_in(a)
        a_out = self.action_in_act(action_residual) + self.residual_flag * action_residual
        for layer in self.action_layer_list:
            a_out = layer(a_out) + self.residual_flag * action_residual
        action_out = self.action_out_layer(a_out)
        return state_out + action_out


class HPRewardNN(torch.nn.Module):

    def __init__(self, state_size, action_size, hps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_hidden = hps["num_hidden"]
        width_hidden = hps["width_hidden"]
        use_bias = hps["use_bias"]

        self.residual_flag = hps["use_res_connection"]
        func = init_func_generator(hps["init_std"])

        self.state_in = torch.nn.Linear(state_size, width_hidden, bias=use_bias).apply(func)
        self.state_in_act = get_random_activation(relu=hps["relu"],
                                            sin=hps["sin"],
                                            tanh=hps["tanh"],
                                            sigmoid=hps["sigmoid"]
                                            )
        self.state_layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["dropout_p"]))
            seq_list.append(get_random_activation(relu=hps["relu"],
                                                  sin=hps["sin"],
                                                  tanh=hps["tanh"],
                                                  sigmoid=hps["sigmoid"]
                                                  ))
            self.state_layer_list.append(torch.nn.Sequential(*seq_list).apply(func))
        self.state_out_layer = torch.nn.Linear(width_hidden, 1, bias=use_bias).apply(func)

        self.a_residual_flag = hps["a_use_res_connection"]
        func = init_func_generator(hps["a_init_std"])

        num_hidden = hps["a_num_hidden"]
        width_hidden = hps["a_width_hidden"]
        use_bias = hps["a_use_bias"]

        self.action_in = torch.nn.Linear(action_size, width_hidden, bias=use_bias).apply(func)
        self.action_in_act = get_random_activation(relu=hps["a_relu"],
                                            sin=hps["a_sin"],
                                            tanh=hps["a_tanh"],
                                            sigmoid=hps["a_sigmoid"]
                                            )
        self.action_layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["a_use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["a_dropout_p"]))
            seq_list.append(get_random_activation(relu=hps["a_relu"],
                                                  sin=hps["a_sin"],
                                                  tanh=hps["a_tanh"],
                                                  sigmoid=hps["a_sigmoid"]
                                                  ))
            self.action_layer_list.append(torch.nn.Sequential(*seq_list).apply(func))
        self.action_out_layer = torch.nn.Linear(width_hidden, 1, bias=use_bias).apply(func)

        self.ns_residual_flag = hps["ns_use_res_connection"]

        func = init_func_generator(hps["ns_init_std"])

        num_hidden = hps["ns_num_hidden"]
        width_hidden = hps["ns_width_hidden"]
        use_bias = hps["ns_use_bias"]

        self.ns_in = torch.nn.Linear(state_size, width_hidden, bias=use_bias).apply(func)
        self.ns_in_act = get_random_activation(relu=hps["ns_relu"],
                                            sin=hps["ns_sin"],
                                            tanh=hps["ns_tanh"],
                                            sigmoid=hps["ns_sigmoid"]
                                            )
        self.ns_layer_list = []
        for i in range(num_hidden):
            seq_list = [torch.nn.Linear(width_hidden, width_hidden, bias=use_bias)]
            if hps["ns_use_dropout"]:
                seq_list.append(CustomFixedDropout(width_hidden, hps["ns_dropout_p"]))
            seq_list.append(get_random_activation(relu=hps["ns_relu"],
                                                  sin=hps["ns_sin"],
                                                  tanh=hps["ns_tanh"],
                                                  sigmoid=hps["ns_sigmoid"]
                                                  ))
            self.ns_layer_list.append(torch.nn.Sequential(*seq_list).apply(func))
        self.ns_out_layer = torch.nn.Linear(width_hidden, 1, bias=use_bias).apply(func)

    def forward(self, s, a, ns):
        state_residual = self.state_in(s)
        s_out = self.state_in_act(state_residual) + self.residual_flag * state_residual
        for layer in self.state_layer_list:
            s_out = layer(s_out) + self.residual_flag * state_residual
        state_out = self.state_out_layer(s_out)

        action_residual = self.action_in(a)
        a_out = self.action_in_act(action_residual) + self.a_residual_flag * action_residual
        for layer in self.action_layer_list:
            a_out = layer(a_out) + self.residual_flag * action_residual
        action_out = self.action_out_layer(a_out)

        ns_residual = self.ns_in(ns)
        ns_out = self.ns_in_act(ns_residual) + self.ns_residual_flag * ns_residual
        for layer in self.ns_layer_list:
            ns_out = layer(ns_out) + self.ns_residual_flag * ns_residual
        next_state_out = self.ns_out_layer(ns_out)
        return state_out + action_out + next_state_out


class MomentumEnv(gym.Env):

    def __init__(self, hps):
        self.state = None

        self.constant_reward = random.random() > 0.5
        self.discrete = False
        if random.random() > 0.5:
            self.action_dim = 1
            self.discrete = True
            dim = np.random.randint(2, high=5)
            self.action_space = spaces.Discrete(dim)
            self.discrete_choices = 6 * (np.random.rand(dim) - 0.5)
        else:
            self.action_dim = np.random.randint(1, high=4)
            max_action = 2.5 * np.random.rand() + 0.5
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(self.action_dim,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 11)
        # maximum of momentum dims are obs_size // 2 min num is 0 is
        self.num_momentum_dims = np.random.randint(0, (self.obs_size//2) + 1)
        self.state_scale = hps["state_scale"] * np.random.rand(self.obs_size - 2 * self.num_momentum_dims)
        self.state_offset = hps["state_offset"] * (np.random.rand() - 0.5)
        self.total_steps = 0

        self.NN_list = []
        for i in range(self.obs_size - 2 * self.num_momentum_dims):
            self.NN_list.append(HPStepNN(self.obs_size + self.action_dim, output_size=1, hps=hps))

        self.pos_list = []
        self.vel_list = []
        for j in range(self.num_momentum_dims):
            dym_type = np.random.choice(["sin", "cos", "x", "y", "rad"])
            self.pos_list.append(PosDym(dym_type))
            self.vel_list.append(VelDym(dym_type, self.action_dim))
        self.reward_model = HPStepNN(2 * self.obs_size + self.action_dim, output_size=1, hps=hps)
        self.eps_steps = 0

    def step(self, action):
        if self.discrete:
            action = self.discrete_choices[action]
        next_state_and_reward = []
        if isinstance(action, int) or isinstance(action, float):
            action = [action]
        else:
            action = list(action)
        state_action = torch.tensor(list(self.state) + action).float()
        with torch.no_grad():
            for g in self.NN_list:
                next_state_and_reward.append(g.forward(state_action).item())
            for v, p in zip(self.vel_list, self.pos_list):
                # first update position and velocity
                v.update(action, p)
                p.update(v)
                # append to next state
                next_state_and_reward.append(p.get_position())
                next_state_and_reward.append(v.get_velocity())
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
        NNstates = (np.random.rand(self.obs_size - 2 * self.num_momentum_dims) - self.state_offset) * self.state_scale
        velocity_position_states = []
        for v, p in zip(self.vel_list, self.pos_list):
            velocity_position_states.append(p.reset())
            velocity_position_states.append(v.reset())
        self.state = np.concatenate((NNstates, velocity_position_states))
        self.eps_steps = 0
        return self.state, None


class VaryMomentumEnv(gym.Env):

    def __init__(self, hps):
        self.state = None

        self.constant_reward = random.random() > 0.5
        self.discrete = False
        if random.random() > 0.5:
            self.action_dim = 1
            self.discrete = True
            dim = np.random.randint(2, high=5)
            self.action_space = spaces.Discrete(dim)
            self.discrete_choices = 6 * (np.random.rand(dim) - 0.5)
        else:
            self.action_dim = np.random.randint(1, high=4)
            max_action = 2.5 * np.random.rand() + 0.5
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(self.action_dim,), dtype=np.float32
            )

        self.obs_size = random.randint(3, 11)
        # maximum of momentum dims are obs_size // 2 min num is 0 is
        self.num_momentum_dims = np.random.randint(0, (self.obs_size//2) + 1)
        self.state_scale = hps["state_scale"] * np.random.rand(self.obs_size - 2 * self.num_momentum_dims)
        self.state_offset = hps["state_offset"] * (np.random.rand() - 0.5)
        self.total_steps = 0

        self.NN_list = []
        for i in range(self.obs_size - 2 * self.num_momentum_dims):
            # TODO sample set of NN hyper parameter
            #cfg = self.generate_nn_config()
            #self.NN_list.append(HPStepNN(self.obs_size + self.action_dim, output_size=1, hps=cfg))
            cfg = self.generate_nn_config_s_a_ns()
            self.NN_list.append(HPStateActionNN(self.obs_size, self.action_dim, hps=cfg))

        self.pos_list = []
        self.vel_list = []
        for j in range(self.num_momentum_dims):
            dym_type = np.random.choice(["sin", "cos", "x", "y", "rad"])
            self.pos_list.append(PosDym(dym_type))
            self.vel_list.append(VelDym(dym_type, self.action_dim))
        # cfg = self.generate_nn_config()
        # self.reward_model = HPStepNN(2 * self.obs_size + self.action_dim, output_size=1, hps=cfg)
        cfg = self.generate_nn_config_s_a_ns()
        self.reward_model = HPRewardNN(self.obs_size, self.action_dim, hps=cfg)
        self.eps_steps = 0

    def generate_nn_config(self):
        config = {}
        config["num_hidden"] = np.random.randint(0, 3)
        config["width_hidden"] = np.random.randint(4, 64)
        config["use_bias"] = np.random.choice([True, False])
        config["use_res_connection"] = np.random.choice([True, False])
        config["use_dropout"] = np.random.choice([True, False])
        config["dropout_p"] = 0.25 * np.random.rand() + 0.75
        config["relu"] = np.random.choice([True, False])
        config["sin"] = np.random.choice([True, False])
        config["tanh"] = np.random.choice([True, False])
        config["sigmoid"] = np.random.choice([True, False])
        config["use_layer_norm"] = False # deprecated
        return config

    def generate_nn_config_s_a_ns(self):
        config = {}
        config["num_hidden"] = np.random.randint(0, 2)
        config["width_hidden"] = np.random.randint(4, 8)
        config["use_bias"] = np.random.choice([True, False])
        config["use_res_connection"] = np.random.choice([True, False])
        config["use_dropout"] = np.random.choice([True, False])
        config["dropout_p"] = 0.25 * np.random.rand() + 0.75
        config["relu"] = np.random.choice([True, False])
        config["sin"] = np.random.choice([True, False])
        config["tanh"] = np.random.choice([True, False])
        config["sigmoid"] = np.random.choice([True, False])
        config["init_std"] = 0.5 * np.random.rand() # TODO find range

        config["a_num_hidden"] = np.random.randint(0, 2)
        config["a_width_hidden"] = np.random.randint(4, 8)
        config["a_use_bias"] = np.random.choice([True, False])
        config["a_use_res_connection"] = np.random.choice([True, False])
        config["a_use_dropout"] = np.random.choice([True, False])
        config["a_dropout_p"] = 0.25 * np.random.rand() + 0.75
        config["a_relu"] = np.random.choice([True, False])
        config["a_sin"] = np.random.choice([True, False])
        config["a_tanh"] = np.random.choice([True, False])
        config["a_sigmoid"] = np.random.choice([True, False])
        config["a_init_std"] = 0.5 * np.random.rand() # TODO find range


        config["ns_num_hidden"] = np.random.randint(0, 2)
        config["ns_width_hidden"] = np.random.randint(4, 8)
        config["ns_use_bias"] = np.random.choice([True, False])
        config["ns_use_res_connection"] = np.random.choice([True, False])
        config["ns_use_dropout"] = np.random.choice([True, False])
        config["ns_dropout_p"] = 0.25 * np.random.rand() + 0.75
        config["ns_relu"] = np.random.choice([True, False])
        config["ns_sin"] = np.random.choice([True, False])
        config["ns_tanh"] = np.random.choice([True, False])
        config["ns_sigmoid"] = np.random.choice([True, False])
        config["ns_init_std"] = 0.5 * np.random.rand() # TODO find range
        return config

    def step(self, action):
        if self.discrete:
            action = self.discrete_choices[action]
        next_state_and_reward = []
        if isinstance(action, int) or isinstance(action, float):
            action = [action]
        else:
            action = list(action)
        #state_action = torch.tensor(list(self.state) + action).float()
        state_tensor = torch.tensor(list(self.state)).float()
        action_tensor = torch.tensor(action).float()
        with torch.no_grad():
            for g in self.NN_list:
                # next_state_and_reward.append(g.forward(state_action).item())
                next_state_and_reward.append(g.forward(state_tensor, action_tensor).item())
            for v, p in zip(self.vel_list, self.pos_list):
                # first update position and velocity
                v.update(action, p)
                p.update(v)
                # append to next state
                next_state_and_reward.append(p.get_position())
                next_state_and_reward.append(v.get_velocity())
            #next_state_and_reward.append(self.reward_model(torch.tensor(list(self.state) + next_state_and_reward + action).float()))
            next_state_and_reward.append(
                self.reward_model(state_tensor, action_tensor, torch.tensor(next_state_and_reward).float()).float())
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
        NNstates = (np.random.rand(self.obs_size - 2 * self.num_momentum_dims) - self.state_offset) * self.state_scale
        velocity_position_states = []
        for v, p in zip(self.vel_list, self.pos_list):
            velocity_position_states.append(p.reset())
            velocity_position_states.append(v.reset())
        self.state = np.concatenate((NNstates, velocity_position_states))
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
    elif env_name == "MomentumEnv":
        env = MomentumEnv(hps)
    elif env_name == "VaryArchitectureMomentumEnv":
        env = VaryMomentumEnv(hps)
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
            act = torch.tensor((3 - len(action)) * [0.] + action)
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
        mean = torch.mean(X[:, b], dim=0)
        std = torch.std(X[:, b], dim=0)
        X[:, b] = torch.nan_to_num((X[:, b] - mean) / std, nan=0)
        mean = torch.mean(Y[:, b], dim=0)
        std = torch.std(Y[:, b], dim=0)
        Y[:, b] = torch.nan_to_num((Y[:, b] - mean) / std, nan=0)
    return X, Y


def TNLU(max_mean, min_mean, minimum, to_round):
    sampled_mean = np.exp(np.random.uniform(np.log(min_mean), np.log(max_mean)))
    sampled_std = np.exp(np.random.uniform(np.log(min_mean), np.log(max_mean)))
    value = np.random.normal(sampled_mean, sampled_std)
    value_trunc = max(value, minimum)
    if to_round:
        return round(value_trunc)
    return value_trunc


class AdditiveNoiseLayer(torch.nn.Module):

    def __init__(self, size, std, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # generate a mask in shape hidden
        self.noise = torch.normal(mean=0., std=std, size=(size,))

    def forward(self, x):
        return x + self.noise


def generate_bnn(in_size, out_size):
    depth = TNLU(6, 1, 2, to_round=True)
    width = TNLU(130, 5, 4, to_round=True)
    additive_noise_std = TNLU(.3, 0.0001, 0.0, to_round=False)  # 0.001 * np.random.rand() + 0.0003  #
    init_std = TNLU(10., 0.01, 0.0, to_round=False)  # 0.1 * np.random.rand() + 0.09  #

    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=init_std)

    use_bias = False # np.random.choice([True, False])
    use_res_connection = True # np.random.choice([True, False])
    act_funct = np.random.choice(
        [torch.nn.Tanh, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ELU, SinActivation, NoOpActivation])
    dropout = np.random.choice([True, False])
    dropout_p = 1 - (0.9 * np.random.beta(np.random.uniform(0.1, 5.0), np.random.uniform(0.1, 5.0)))
    bnn_model = BNN(in_size, out_size, depth, width, use_bias, use_res_connection, act_funct, dropout, dropout_p,
                    additive_noise_std)
    bnn_model.apply(weight_init)
    return bnn_model


class BNN(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 depth,
                 width,
                 use_bias,
                 use_res_conn,
                 activation,
                 dropout,
                 dropout_p,
                 additive_noise_std,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_flag = use_res_conn
        self.in_lin = torch.nn.Linear(input_size, width, bias=use_bias)
        self.in_act = activation()
        layer_list = []
        for i in range(depth):
            seq_list = [torch.nn.Linear(width, width, bias=use_bias)]
            if dropout:
                seq_list.append(CustomFixedDropout(width, dropout_p))
            seq_list.append(AdditiveNoiseLayer(width, additive_noise_std))
            seq_list.append(activation())
            layer_list.append(torch.nn.Sequential(*seq_list))
        self.layer_list = torch.nn.ModuleList(layer_list)
        self.out_lin = torch.nn.Linear(width, output_size, bias=use_bias)

    def forward(self, x):
        residual = self.in_lin(x)
        out = self.in_act(residual)
        for layer in self.layer_list:
            out = layer(out) + self.residual_flag * residual
        return self.out_lin(out)


def get_bnn_train_batch(seq_len, batch_size, num_features, hyperparameters):
    # generate input data
    X = torch.rand((seq_len, batch_size, num_features))
    Y = torch.Tensor()
    for b in range(batch_size):
        # sample random state dim and action dim
        state_dim = np.random.randint(3, 12)
        action_dim = np.random.randint(1, 4)

        # zero out not used action dims
        X[:, b, state_dim:-3] = 0
        X[:, b, num_features-3+action_dim:] = 0

        # sample BNN for state dym
        state_dynamics_bnn = generate_bnn(state_dim + action_dim, state_dim)
        # forward state dym BNN
        # TODO improve representation of action
        next_state = state_dynamics_bnn(
            torch.cat((X[:, b:b+1, :state_dim], X[:, b:b+1, num_features - 3:num_features - 3 + action_dim]), dim=2))

        # sample BNN for reward
        reward_dynamics_bnn = generate_bnn(2 * state_dim + action_dim, 1)

        # forward BNN for reward
        reward = reward_dynamics_bnn(
            torch.cat((X[:, b:b + 1, :state_dim], X[:, b:b + 1, num_features - 3:num_features - 3 + action_dim],
                       next_state), dim=2))


        final_total_dym = torch.cat(
            (next_state, torch.zeros((seq_len, 1, num_features - 1 - state_dim)), reward), dim=2)

        # Fill Output
        Y = torch.cat((Y, final_total_dym), dim=1)

        # shuffle zero dims
        state_per_dims = torch.randperm(num_features-3)
        X[:, b:b+1, :num_features-3] = X[:, b:b+1, :num_features-3][:, :, state_per_dims]

        Y[:, b:b+1, :num_features-3] = Y[:, b:b+1, :num_features-3][:, :, state_per_dims]

        action_per_dim = torch.randperm(3)
        X[:, b:b+1, num_features-3:] = X[:, b:b+1, num_features-3:][:, :, action_per_dim]

    # 0 mean 1 variacne Normalize
    x_means = torch.mean(X, dim=0)
    x_stds = torch.std(X, dim=0)
    X = torch.nan_to_num((X - x_means) / x_stds, nan=0)

    # min max scaling
    """
    y_min = Y.min(dim=0, keepdim=True).values.min(dim=0, keepdim=True).values
    y_max = Y.max(dim=0, keepdim=True).values.max(dim=0, keepdim=True).values
    Y = torch.nan_to_num((Y - y_min)/(y_max - y_min))
    """

    # 0 mean 1 variacne Normalize
    y_means = torch.mean(Y, dim=0)
    y_stds = torch.std(Y, dim=0)
    Y = torch.nan_to_num((Y - y_means) / y_stds, nan=0)
    #Y = torch.where(Y > 100., torch.log(Y) + 100, Y)
    #Y = torch.where(Y < -100., torch.exp(Y) - 100, Y)
    return X, Y


def get_bnn_train_seq(seq_len, batch_size, num_features, hyperparameters):
    # generate input data
    X = torch.zeros((seq_len, batch_size, num_features))
    Y = torch.zeros((seq_len, batch_size, num_features))
    for b in range(batch_size):
        # sample random state dim and action dim
        state_dim = np.random.randint(3, 12)
        action_dim = np.random.randint(1, 4)
        # sample BNN for state dym
        state_dynamics_bnn = generate_bnn(state_dim + action_dim, state_dim)

        # sample BNN for reward
        reward_dynamics_bnn = generate_bnn(2 * state_dim + action_dim, 1)

        obs = torch.rand(state_dim)
        for i in range(seq_len):
            action = torch.rand(action_dim)
            ns = state_dynamics_bnn(torch.cat((obs, action)))
            r = reward_dynamics_bnn(torch.cat((obs, ns, action)))

            X[i, b, :state_dim] = obs
            X[i, b, 11:11+action_dim] = action

            Y[i, b, :state_dim] = ns
            Y[i, b, -1] = r

            obs = ns

            if i % 50 == 0:
                obs = torch.rand(state_dim)

        # shuffle zero dims
        state_per_dims = torch.randperm(num_features - 3)
        X[:, b:b + 1, :num_features - 3] = X[:, b:b + 1, :num_features - 3][:, :, state_per_dims]

        Y[:, b:b + 1, :num_features - 3] = Y[:, b:b + 1, :num_features - 3][:, :, state_per_dims]

        action_per_dim = torch.randperm(3)
        X[:, b:b + 1, num_features - 3:] = X[:, b:b + 1, num_features - 3:][:, :, action_per_dim]

    # 0 mean 1 variacne Normalize
    x_means = torch.mean(X, dim=0)
    x_stds = torch.std(X, dim=0)
    X = torch.nan_to_num((X - x_means) / x_stds, nan=0)


    # 0 mean 1 variacne Normalize
    y_means = torch.mean(Y, dim=0)
    y_stds = torch.std(Y, dim=0)
    Y = torch.nan_to_num((Y - y_means) / y_stds, nan=0)
    return X, Y


@torch.no_grad()
def get_bnn_batch(
        batch_size,
        seq_len,
        num_features,
        device=default_device,
        hyperparameters=None,
        **kwargs
):

    X, Y = get_bnn_train_batch(seq_len, batch_size, num_features, hyperparameters)
    # print(Y.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values)
    # print(Y.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values)
    return Batch(x=X, y=Y, target_y=Y)


@torch.no_grad()
def get_bnn_sequantial_batch(
        batch_size,
        seq_len,
        num_features,
        device=default_device,
        hyperparameters=None,
        **kwargs
):

    X, Y = get_bnn_train_seq(seq_len, batch_size, num_features, hyperparameters)
    return Batch(x=X, y=Y, target_y=Y)