import numpy as np
import matplotlib.pyplot as plt


environments = ["Reacher-v4", "CartPole-v1", "Pendulum-v1", "SimpleEnv"]
for e in environments:
    data = np.load(f"{e}_mag.npy")
    mag = data[0, :]
    norm_mag = data[1, :]
    loss = data[2, :]

    plt.bar(norm_mag, loss, width=0.01)
    plt.title("Loss wrt Magnitude between input and target for " + e)
    plt.xlabel("Magnitude")
    plt.ylabel("Loss")
    plt.savefig(e + "_input_target_mag.png", dpi=300)
    plt.cla()
    plt.clf()

    plt.bar(mag, loss, width=0.01)
    plt.title("Loss wrt Magnitude between state and next state for " + e)
    plt.xlabel("Magnitude")
    plt.ylabel("Loss")
    plt.savefig(e + "_s_ns_mag.png", dpi=300)
    plt.cla()
    plt.clf()
