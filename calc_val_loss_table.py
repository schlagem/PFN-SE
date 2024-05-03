from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from train import build_model
import encoders
import os
import gymnasium as gym
from simple_env import SimpleEnv
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_policy(obs_size, a_size, hidden_size=64):
    model = nn.Sequential(nn.Linear(obs_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, hidden_size),
                          nn.ReLU(),
                          nn.Linear(hidden_size, a_size)
                          )
    return model


# TODO summarize this function somewhere (3/3)
def get_environment(env_name):
    if env_name == "SimpleEnv":
        return SimpleEnv()
    else:
        env = gym.make(env_name)
        return env


def get_transitions_dir(env_name, random_fraction):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    val_path = os.path.join(dir_path, "val_transitions")
    # Make directory for env if not existing
    env_path = os.path.join(val_path, env_name)
    # Make directory for random fraction if not existing
    fraction_str = "expert_" + str(int(100*(1-random_fraction))) + "-" + "random_" + str(int(100*random_fraction))
    fraction_path = os.path.join(env_path, fraction_str)
    return fraction_path


def gather_context(length, batch_size, feature_size, env_name, action_gather_type="random"):
    """
    Gather Transition of episodes and add to context used for prediction.
    x_context (shape: seq_len, batch, num_features) contains observation and action
    y_context (shape: seq_len, batch, num_features) contains new observation and reward
    x_mean (shape: batch, num_features) is mean of x per column (first seq_len - 1 elements)
    x_std (shape: batch, num_features) is std deviation of x per column (first seq_len - 1 elements)
    y_mean (shape: batch, num_features) is mean of x per column (first seq_len - 1 elements)
    y_std (shape: batch, num_features) is std deviation of y per column (first seq_len - 1 elements)
    """
    x_context = torch.zeros((length, batch_size, feature_size))
    y_context = torch.zeros((length, batch_size, feature_size))

    for b in range(batch_size):
        env = get_environment(env_name)
        observation, info = env.reset()
        if action_gather_type == "Random Policy":
            if type(env.action_space) is gym.spaces.Discrete:  # Discrete case
                a_shape = env.action_space.n
            else:  # Continous
                a_shape = env.action_space.shape[0]
            policy = get_random_policy(env.observation_space.shape[0], a_shape)
        repeats = 100
        for k in range(length):
            if action_gather_type == "random":
                action = env.action_space.sample()
            elif action_gather_type == "repeats":
                if repeats >= 3:
                    action = env.action_space.sample()
                    repeats = 0
                else:
                    repeats += 1
            elif action_gather_type == "Random Policy":
                with torch.no_grad():
                    if type(env.action_space) is gym.spaces.Discrete:  # discrete case
                        # TODO shoudl this be torch.long
                        action = torch.argmax(policy(torch.tensor(observation, dtype=torch.float)))
                    else:  # Continous case
                        action = policy(torch.tensor(observation, dtype=torch.float))
                action = action.numpy()
            else:
                raise NotImplementedError("No Valid Context gathering method used!")
            x_context[k, b, :observation.shape[0]] = torch.tensor(observation)
            a_shape = 1 if type(action) is int or len(action.shape) == 0 else action.shape[0]
            x_context[k, b, -a_shape:] = torch.tensor(action)
            observation, reward, terminated, truncated, info = env.step(action)

            y_context[k, b, :observation.shape[0]] = torch.tensor(observation)
            y_context[k, b, -1] = torch.tensor(reward)
            if terminated or truncated:
                repeats = 100
                observation, info = env.reset()

    x_mean = torch.mean(x_context[:1000, :, :], dim=0)
    x_std = torch.std(x_context[:1000, :, :], dim=0)
    x_context = torch.nan_to_num((x_context - x_mean) / x_std, nan=0)

    y_mean = torch.mean(y_context[:1000, :, :], dim=0)
    y_std = torch.std(y_context[:1000, :, :], dim=0)
    y_context = torch.nan_to_num((y_context - y_mean) / y_std, nan=0)

    perm = torch.randperm(length)
    x_context = x_context[perm]
    y_context = y_context[perm]

    return x_context.to(device), y_context.to(device), x_mean.to(device), x_std.to(device), y_mean.to(device), y_std.to(device)


def val_loss_table(model, debug_truncation=False):
    # Number of states plus number of action -> maximum size
    num_features = 14
    number_context_batches = 8

    summary_dict = {}

    env_collection = ["CartPole-v1", "Pendulum-v1", "SimpleEnv", "Reacher-v4"]
    random_fractions = [1.0, 0.5, 0.0]
    for environment in env_collection:
        summary_dict[environment] = {}
        x, y, x_means, x_stds, y_means, y_stds = gather_context(1001, number_context_batches, num_features, environment,
                                                                action_gather_type="random")
        train_len = 1000
        train_x = x[:train_len]
        train_y = y[:train_len]
        test_x = x[:]
        for fraction in random_fractions:
            # Logging of validations
            losses = []
            axis_losses = []
            context_losses = []
            # get dir of env + random action fraction
            transition_dir = get_transitions_dir(environment, fraction)

            # load transitions for this setting
            states = np.load(os.path.join(transition_dir, "states.npy"))
            actions = np.load(os.path.join(transition_dir, "actions.npy"))
            next_states = np.load(os.path.join(transition_dir, "next_states.npy"))
            rewards = np.load(os.path.join(transition_dir, "rewards.npy"))
            dones = np.load(os.path.join(transition_dir, "dones.npy"))

            # for each transition predict using #batch_size different context
            for i, transition in enumerate(tqdm(zip(states, actions, next_states, rewards, dones), total=states.shape[0])):
                s, a, ns, r, d = transition
                if debug_truncation and i > 5:
                    break
                new_x = torch.zeros_like(test_x[1000, 0])
                new_x[:s.shape[0]] = torch.tensor(s)
                # scalar actions are not saved as 1-d arrays
                action_shape = 1 if len(a.shape) == 0 else a.shape[0]
                new_x[-action_shape:] = torch.tensor(a)
                norm_x = torch.nan_to_num((new_x - x_means) / x_stds, nan=0)
                test_x[1000] = norm_x

                new_y = torch.zeros_like(y[1000, 0])
                new_y[:ns.shape[0]] = torch.tensor(ns)
                new_y[-1] = torch.tensor(r)

                norm_y = torch.nan_to_num((new_y - y_means) / y_stds, nan=0)
                y[1000] = norm_y


                with torch.no_grad():
                    # predicting new logits
                    logits = model(train_x, train_y, test_x)
                    # from normalized values to true values
                    y_pred = (logits * y_stds) + y_means
                    y_target = (y * y_stds) + y_means
                    # loss average over all axis
                    loss = torch.nn.functional.mse_loss(y_pred[1000:, :, :], y_target[1000:, :, :], reduction="none")
                    # Total mean loss
                    losses.append(loss.mean())
                    # loss per axis
                    axis_losses.append(loss.mean(axis=(0, 1)))
                    # loss for each context to measure variance between context
                    context_losses.append(loss.mean(axis=(0, 2)))

            single_setting_dict = {
                # Overall loss of all axis context and steps
                "loss": torch.tensor(losses).mean().item(),
                # std of the losses
                "loss_std": torch.tensor(losses).std().item(),
                # losses per axis -> dim 1 of states .... and reward prediction (over all axis and steps)
                "axis_loss": torch.cat(axis_losses).view(-1, num_features).mean(axis=0).tolist(),
                # std for the per axis loss
                "axis_loss_std": torch.cat(axis_losses).view(-1, num_features).std(dim=0).tolist(),
                # Loss per context over all axis and all steps
                "context_loss": torch.cat(context_losses).view(-1, number_context_batches).mean(axis=0).tolist(),
                # std per context over all axis and steps
                "context_loss_std": torch.cat(context_losses).view(-1, number_context_batches).std(dim=0).tolist(),
                # std between per context loss -> variation of means of different contexts
                "std_between_context": torch.cat(context_losses).view(-1, number_context_batches).mean(axis=0).std().tolist(),
            }
            summary_dict[environment][str(fraction)] = single_setting_dict
    return summary_dict


def get_model():
    num_features = 14
    # Loss to be used for evaluation of model
    criterion = nn.MSELoss(reduction='none')

    # building Transformer model and loading weights
    hps = {'test': True}
    pfn = build_model(
        criterion=criterion,
        encoder_generator=encoders.Linear,
        test_batch=None,
        n_out=14,
        emsize=512, nhead=4, nhid=1024, nlayers=6,
        seq_len=1001,
        y_encoder_generator=encoders.Linear,
        decoder_dict={},
        extra_prior_kwargs_dict={'num_features': num_features, 'hyperparameters': hps},
    )
    print(
        f"Using a Transformer with {sum(p.numel() for p in pfn.parameters()) / 1000 / 1000:.{2}f} M parameters"
    )
    pfn.load_state_dict(torch.load("saved_models/first_incumbent.pt"))
    pfn.eval()
    return pfn.to(device)


if __name__ == '__main__':
    results = val_loss_table(get_model())
    print(results)
    exit()
    # TODO change save name to show checkpoint name
    with open('val_transitions/scores/validation_scores.json', 'w') as fp:
        json.dump(results, fp, indent=4)

