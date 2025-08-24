import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroSample


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # Define weight as a PyroSample (distribution, not Parameter)
        self.weight = PyroSample(
            dist.Normal(
                torch.zeros(out_features, in_features, device=device),
                torch.ones(out_features, in_features, device=device)
            ).to_event(2)
        )
        # Define bias as a PyroSample
        self.bias = PyroSample(
            dist.Normal(
                torch.zeros(out_features, device=device),
                torch.ones(out_features, device=device)
            ).to_event(1)
        )

    def forward(self, x):
        # At forward time, F.linear will sample weight & bias
        return F.linear(x, self.weight, self.bias)


class BayesianHead(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, device="cpu"):
        super().__init__()
        self.fc1 = BayesianLinear(in_dim, hidden, device=device)
        self.fc2 = BayesianLinear(hidden, out_dim, device=device)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)


class BayesianDBN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, device="cpu"):
        super().__init__()
        self.head = BayesianHead(in_dim, hidden, out_dim, device)

    def forward(self, x, y=None):
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        return self.head(x)
