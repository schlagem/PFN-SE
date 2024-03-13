import matplotlib.pyplot as plt
import torch


titles = ["Cart Position", "Cart Position", "Pole Angle", "Pole Angular Velocity", "Reward"]

values_pfn = torch.load("states_visited_cartpole_PFN.pt")
values_real = torch.load("states_visited_cartpole_real_1.pt")

print(values_pfn.shape)
print(values_real.shape)
fig, axs = plt.subplots(1, 5)
fig.tight_layout()
for k, ax in enumerate(axs.reshape(-1)):
    ax.set_box_aspect(1)
    ax.hist(values_pfn[:, k].view(-1), 50, density=False, label="PFN Values")
    ax.hist(values_real[:, k].view(-1), 50, density=False, alpha=0.5, label="Real Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(titles[k])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right')
plt.show()