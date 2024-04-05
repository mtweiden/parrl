"""Based on: https://arxiv.org/pdf/2403.03950.pdf"""
from torch import sum
from torch.special import erf
from torch import Tensor
import torch.special
import torch.nn as nn
import torch.nn.functional as F


class HLGaussLoss(nn.Module):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float,
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(min_value, max_value, num_bins + 1)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: Tensor) -> Tensor:
        _target = self.support - target.unsqueeze(-1)
        _scale = (2 ** 0.5) * self.sigma
        cdf_evals = erf(_target / _scale)
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        probs = bin_probs / z.unsqueeze(-1)
        return probs

    def transform_from_probs(self, probs: Tensor) -> Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        mean = sum(probs * centers, dim=-1)
        return mean
