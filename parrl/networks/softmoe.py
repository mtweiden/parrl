"""
Soft Mixture of Experts.
"""
from einops import einsum
from einops import rearrange

from torch import nn
from torch import randn
from torch import stack
from torch import Tensor
from torch.nn.functional import softmax


class SoftMoELayer(nn.Module):

    def __init__(
        self,
        latent_dim: int,
        expert_dim: int,
        num_experts: int,
        slots_per_expert: int,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert

        init_weights = randn(latent_dim, num_experts, slots_per_expert)
        init_weights = init_weights / latent_dim ** 0.5
        self.phi = nn.Parameter(init_weights)

        layer = nn.Sequential(
            nn.Linear(latent_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, latent_dim),
        )
        self.experts = nn.ModuleList([layer for _ in range(num_experts)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Notes:
            - b: batch size
            - d: latent dim
            - m: number of tokens
            - n: number experts
            - p: slots per expert
        """
        logits = einsum(x, self.phi, 'b m d, d n p -> b m n p')
        # Dispatch weights
        d = softmax(logits, dim=1)
        # Combination weights
        logits = rearrange(logits, 'b m n p -> b m (n p)')
        c = softmax(logits, dim=-1)
        c = rearrange(c, 'b m (n p) -> b m n p', n=self.num_experts)

        x_i = einsum(x, d, 'b m d, b m n p -> b n p d')
        y_i = stack(
            [expert(x_i[:, i, :, :]) for i, expert in enumerate(self.experts)],
            dim=1,
        )
        # Combine expert results
        y = einsum(y_i, c, 'b n p d, b m n p -> b m d')
        return y
