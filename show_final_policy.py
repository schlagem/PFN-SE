import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


n_actions = 2
n_observations = 4

policy_net = DQN(n_observations, n_actions)
policy_net.load_state_dict(torch.load("policy_trained_on_se.pt"))
policy_net.eval()

real_env = gym.make("CartPole-v1", render_mode="human")

observation, _ = real_env.reset()
for d in range(10):
    for k in range(200):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        act = policy_net(state).max(1)[1].view(1, 1)
        observation, reward, terminated, truncated, _ = real_env.step(act.item())
        real_env.render()
        if terminated or k > 199:
            print(k)
            real_env.reset()
            break
