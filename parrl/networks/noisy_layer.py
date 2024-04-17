"""Based off NoiseyNet: https://arxiv.org/pdf/1706.10295.pdf"""
from numpy import sqrt

from torch import nn
from torch import randn
from torch import Tensor
import torch.nn.functional as F


class NoisyLinear(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Tensor

    """
    A Noisy Net version of a pytorch nn.Linear layer.
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        norm = 1 / sqrt(self.in_features)
        self.std_init = 0.5 * norm
        self.mean_init = norm

        weight_size = (self.out_features, self.in_features)
        self.weight_mean = nn.Parameter(Tensor(size=weight_size))
        self.weight_std = nn.Parameter(Tensor(size=weight_size))
        self.register_buffer('weight_noise', Tensor(size=weight_size))

        bias_size = (self.out_features, )
        self.bias_mean = nn.Parameter(Tensor(size=bias_size))
        self.bias_std = nn.Parameter(Tensor(size=bias_size))
        self.register_buffer('bias_noise', Tensor(size=bias_size))

        self.reset_parameters()
        self.reset_noise()

        self.weight = self.weight_mean
        self.bias = self.bias_mean

    def reset_parameters(self) -> None:
        self.weight_mean.data.uniform_(-self.mean_init, self.mean_init)
        self.weight_std.data.fill_(self.std_init)
        self.bias_mean.data.uniform_(-self.mean_init, self.mean_init)
        self.bias_std.data.fill_(self.std_init)

    @staticmethod
    def _f_noise(epsilon: Tensor) -> Tensor:
        """Paper suggests scaling noise."""
        return epsilon.sign() * epsilon.abs().sqrt()

    def reset_noise(self) -> None:
        eps_a = self._f_noise(randn((self.in_features,)))
        eps_b = self._f_noise(randn((self.out_features,)))
        self.weight_noise.copy_(eps_b.ger(eps_a))
        self.bias_noise.copy_(eps_b)


    def forward(self, x: Tensor) -> Tensor:
        w = self.weight_mean + self.weight_std * self.weight_noise
        b = self.bias_mean + self.bias_std * self.bias_noise
        y = F.linear(
            input=x,
            weight=w,
            bias=b,
        )
        return y
