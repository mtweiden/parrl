from typing import Any

from torch import load
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
    
    def load_state(self, state_path_or_dict: str | dict[str, Tensor]) -> None:
        if isinstance(state_path_or_dict, str):
            state_dict = load(state_path_or_dict, map_location='cpu')
        else:
            state_dict = state_path_or_dict
        self.load_state_dict(state_dict)