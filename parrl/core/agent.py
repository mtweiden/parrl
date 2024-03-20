from typing import Any

from torch import nn
from torch import Tensor


class Agent(nn.Module):

    def __init__(
        self,
        discount: float,
    ) -> None:
        super().__init__()
        self.discount = discount

    def forward(self, x: Tensor) -> Any:
        return self.agent_forward(x)

    def agent_forward(self, x: Tensor) -> Any:
        raise NotImplementedError()
    
    def get_action(self, x: Tensor) -> Tensor:
        raise NotImplementedError()