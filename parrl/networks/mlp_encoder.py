from typing import Sequence
    
from torch import nn
from torch import Tensor


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_dims: Sequence[int],
        dropout: float = 0.05,
    ) -> None:
        """
        An MLP encoder for Unitaries that uses a NeRF positional embedding.

        Args:
            input_dim (int): The size of the input.

            layer_dims (Sequence[int]): The dimension of the each layer
                in the model.

            dropout (float): Model dropout. (Default: 0.05)
        """
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        layer_dims = layer_dims
        self.latent_dim = layer_dims[-1]

        self.layer_dims = layer_dims
        if self.layer_dims[0] != self.input_dim:
            self.layer_dims = [self.input_dim] + self.layer_dims
        
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
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x