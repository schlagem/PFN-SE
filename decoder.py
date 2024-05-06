import torch
from torch import nn


class DecoderModel(nn.Module):
    def __init__(self, ninp, nhid, nout):
        super().__init__()
        self.state_decoder = nn.Sequential(
                        nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, nout-3)
                    )
        self.reward_decoder = nn.Sequential(
                        nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, 1)
                    )

    def forward(self, x):
        state_decoded = self.state_decoder(x)
        r_decoded = self.reward_decoder(x)
        zero_shape = (x.shape[0], x.shape[1], 2)
        zero_padding = torch.full(zero_shape, 0.)
        x = torch.cat((state_decoded, zero_padding, r_decoded), dim=2)
        return x


if __name__ == '__main__':
    d = DecoderModel(512, 1024, 14)
    innp = torch.rand(1001, 4, 512)
    out = d(innp)
    print(out.shape)
    print(out[1000, 0, :])