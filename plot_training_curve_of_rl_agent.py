import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_agent_on_pfn import generate_log_dir_path
import os
import json


def read_json(path):
    rewards = []
    time_step = []
    with open(os.path.join(path, "progress.json")) as f:
        for line in f:
            data = json.loads(line)
            rewards.append(data.get["rollout/ep_rew_mean"])
            time_step.append(data.get["time/total_timesteps"])
    return rewards, time_step

def plot_agent_trained_on_oswm(env_name):
    means = []
    time_steps = None
    for seed in range(1, 4):
        path = generate_log_dir_path(env_name, seed)
        evals = np.load(os.path.join(path, "evaluations.npz"))
        read_json(path)
        time_steps = evals["timesteps"]
        means.append(evals["results"].mean(axis=1))

    means = np.array(means)
    plt.plot(time_steps, means.mean(axis=0), label="Mean Test reward")
    plt.fill_between(time_steps,
                     means.mean(axis=0) - means.std(axis=0),
                     means.mean(axis=0) + means.std(axis=0),
                     color='tab:blue', alpha=0.15)
    plt.legend()
    plt.title("Training Curve of RL Agent for " + env_name)
    plt.ylabel("Episode reward")
    plt.xlabel("Time steps")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)

    args = parser.parse_args()
    plot_agent_trained_on_oswm(env_name=args.env)
