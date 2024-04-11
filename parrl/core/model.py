from einops import repeat

from torch import arange
from torch import Tensor


class Model:
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
    
    def take_action(self, state: Tensor, action: Tensor) -> Tensor:
        raise NotImplementedError()

    def take_all_actions(self, state: Tensor) -> Tensor:
        """
        Returns the result of taking all actions on a state. For an input
        of shape (n, d), the output is of shape (a, n, d).
        """
        all_actions = arange(self.num_actions)
        all_states = repeat(state, '... -> a ...', a=self.num_actions)
        all_next_states = self.take_action(all_states, all_actions)
        return all_next_states

    def __call__(self, state: Tensor) -> Tensor:
        return self.take_all_actions(state)
