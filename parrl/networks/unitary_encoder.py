from typing import Sequence
    
from einops import rearrange
from einops import repeat

from torch import atan2
from torch import cos
from torch import nn
from torch import sin
from torch import stack
from torch import tensor
from torch import Tensor

from numpy import pi


class UnitaryEncoder(nn.Module):
    def __init__(
        self,
        num_qubits: int,
        layer_dims: Sequence[int],
        nerf_dim: int = 0,
        dropout: float = 0.05,
    ) -> None:
        """
        An MLP encoder for Unitaries that uses a NeRF positional embedding.

        Args:
            num_qubits (int): The size of unitaries. Determines the size of
                inputs to the network.

            layer_dims (Sequence[int]): The dimension of the each layer
                in the model.

            nerf_dim (int): The dimension of the NeRF positional embedding.
                If 0, then the NeRF positional embedding is not used.
                (Default: 0)

            dropout (float): Model dropout. (Default: 0.05)
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.nerf_dim = nerf_dim
        self.input_dim = 2 * (2 ** self.num_qubits) ** 2 * (2 * self.nerf_dim)
        self.dropout = dropout
        layer_dims = layer_dims
        self.latent_dim = layer_dims[-1]

        self.layer_dims = layer_dims
        if self.layer_dims[0] != self.input_dim:
            self.layer_dims = [self.input_dim] + self.layer_dims
        
        if self.nerf_dim > 0:
            self.nerf = NeRFEmbedding(self.nerf_dim)
        else:
            self.nerf = None

        self.layers = nn.ModuleList([])
        for i in range(len(self.layer_dims) - 1):
            in_dim, out_dim = self.layer_dims[i], self.layer_dims[i + 1]
            self.layers .append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(p=self.dropout),
                )
            )

    @property
    def device(self) -> str:
        return self.layers[0].linear0.weight.device
    
    def dephase(self, x: Tensor) -> Tensor:
        n, c, d = x.shape
        assert c == 2
        # Compute mean of each batch element
        x_r, x_i = x[:, 0], x[:, 1]
        m_r, m_i = x_r.mean(dim=-1), x_i.mean(dim=-1)

        ang = atan2(m_i, m_r + (m_r == 0) * 1e-6)
        ang_conj = -ang
        ang_r, ang_i = cos(ang_conj).unsqueeze(-1), sin(ang_conj).unsqueeze(-1)

        y_r = x_r * ang_r - x_i * ang_i
        y_i = x_r * ang_i + x_i * ang_r

        y = stack([y_r, y_i], dim=1)
        assert y.shape == (n, c, d)
        return y

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = rearrange(x, 'n c x y -> n c (x y)')
        x = self.dephase(x)
        x = rearrange(x, 'n c d -> n (c d)')
        if self.nerf:
            x = self.nerf(x)
            x = rearrange(x, 'n d l -> n (d l)')
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class NeRFEmbedding(nn.Module):
    """Does NeRF embeddings for input Tensors."""
    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L
        self._emb_vec = tensor(
            [pi * 2 ** i for i in range(L)], requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        b, d = x.shape
        if self._emb_vec.device != x.device:
            self._emb_vec = self._emb_vec.to(x.device)
        l = self.L
        x = repeat(x, 'b d -> b d l', l=l)
        _sin = (x * self._emb_vec).sin()
        _cos = (x * self._emb_vec).cos()
        interleaved = stack([_sin, _cos], dim=-1).view(b, d, 2 * l)
        return interleaved