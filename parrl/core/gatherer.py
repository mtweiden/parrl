from typing import Any
from typing import Optional

from abc import ABC
from abc import abstractmethod

from gymnasium import Env

from torch import Tensor

from parrl.core.agent import Agent


class Gatherer(ABC):
    """
    A Gatherer should be launched as a ray actor. This can be done by calling
    `ray.remote(num_cpus=1)(gatherer)`.
    """
    def __init__(
        self,
        agent: Agent,
        env: Env,
        steps_per_iteration: int,
        steps_per_episode: Optional[int] = None,
    ) -> None:
        """
        Abstract Gatherer constructor.

        Args:
            agent (Agent): The agent to use for gathering experience.

            env (Env): The environment to interact with.

            steps_per_iteration (int): The number of steps to gather from the
                environment in a single iteration.

            steps_per_episode (Optional[int]): The number of steps to gather
                from the environment in a single episode. Multiple episodes can
                be run per iteration. If not specified, one episode is run per
                iteration.
        """
        self._agent = agent.cpu()
        self.env = env
        self.steps_per_iteration = steps_per_iteration
        if steps_per_episode is None:
            steps_per_episode = steps_per_iteration
        else:
            if steps_per_episode > steps_per_iteration:
                m = (
                    f'steps_per_episode ({steps_per_episode}) must be <= '
                    f'steps_per_iteration ({steps_per_iteration}).'
                )
                raise ValueError(m)
        self.steps_per_episode = steps_per_episode
    
    def get_agent(self) -> Agent:
        return self._agent
    
    def update_parameters(self, state_dict: dict[str, Tensor]) -> None:
        self._agent.load_state_dict(state_dict)
    
    def update_env_attributes(self, kwargs: dict[str, Any]) -> None:
        for attribute_name, value in kwargs.items():
            try:
                setattr(self.env, attribute_name, value)
            except AttributeError:
                m = f'Attribute {attribute_name} not found in environment.'
                raise AttributeError(m)
    
    @abstractmethod
    def gather(self) -> dict[str, Any]:
        """
        Gather exiperience from the environment using the current agent.

        Returns:
            (dict[str, Any]): A dict training data and statistics. The dict
                should have two keys: 'data' and 'stats'.
        """