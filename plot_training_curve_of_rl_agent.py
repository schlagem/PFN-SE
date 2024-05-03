import numpy as np
import matplotlib.pyplot as plt
import argparse
from train_agent_on_pfn import generate_log_dir_path
import os
import json
import pandas
from itertools import accumulate


def read_json(path):
    rewards = []
    time_step = []
    with open(os.path.join(path, "progress.json")) as f:
        for line in f:
            data = json.loads(line)
            if "rollout/ep_rew_mean" in data.keys():
                rewards.append(data.get("rollout/ep_rew_mean"))
                time_step.append(data.get("time/total_timesteps"))
    return rewards, time_step


def plot_agent_trained_on_oswm(env_name):
    means = []
    train_means = []
    time_steps = None
    train_time_steps = []
    for seed in range(1, 2):
        path = generate_log_dir_path(env_name, seed)
        evals = np.load(os.path.join(path, "evaluations.npz"))
        df = pandas.read_csv(os.path.join(path, "monitor.csv"), header=[1])
        train_means.append(df["r"].tolist())
        train_time_steps.append(list(accumulate(df["l"].tolist())))
        time_steps = evals["timesteps"]
        means.append(evals["results"].mean(axis=1))

    means = np.array(means)
    # train_means = np.array(train_means)

    plt.plot(time_steps, means.mean(axis=0), label="Mean Test reward")
    plt.fill_between(time_steps,
                     means.mean(axis=0) - means.std(axis=0),
                     means.mean(axis=0) + means.std(axis=0),
                     color='tab:blue', alpha=0.15)

    #for m, t, in zip(train_means, train_time_steps):
    #    plt.plot(t, m)
    """
    plt.plot(train_time_steps, train_means.mean(axis=0), label="Mean Test reward")
    plt.fill_between(train_time_steps,
                     train_means.mean(axis=0) - train_means.std(axis=0),
                     train_means.mean(axis=0) + train_means.std(axis=0),
                     color="tab:orange", alpha=0.15)
    """

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
