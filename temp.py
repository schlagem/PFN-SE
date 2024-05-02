import torch
from priors import rl_prior
import argparse



parser = argparse.ArgumentParser(description="test")
parser.add_argument('--no_norm', type=bool, help='To norm the X and Y in OSWM training or not.', default=False)

args = parser.parse_args()
print(args.no_norm)
exit()

hps = {"env_name": "NNEnv", "num_hidden": 1, "relu": False, "sigmoid": False, "sin": True,
       "state_offset": 3.2802608490289904, "state_scale": 18.147661409701062, "tanh": True, "test": False,
       "use_bias": False, "use_dropout": False, "use_layer_norm": True, "use_res_connection": True, "width_hidden": 16,
       "no_norm": True}


seq_len = 1001
batch_size = 4
num_features = 14
X = torch.full((seq_len, batch_size, num_features), 0.)
Y = torch.full((seq_len, batch_size, num_features), float(0.))

for i in range(10):
    print(i)
    X, Y = rl_prior.get_train_batch(seq_len, batch_size, num_features, X, Y, hps)
    for b in range(batch_size):
        mean = torch.mean(X[:, b], dim=0)
        std = torch.std(X[:, b], dim=0)
        print(mean, std)

