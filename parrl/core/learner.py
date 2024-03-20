from typing import Optional 

from abc import ABC
from abc import abstractmethod

from gymnasium import Env

from torch import save
from torch import Tensor

from parrl.core.agent import Agent
from parrl.core.buffer import ReplayBuffer
from parrl.core.gatherer import Gatherer


class Learner(ABC):
    """
    A Learner queries parallel Gatherers for experience and updates an agent.

    The `learn` method of a Learner must be implemented to handle details of
    individual RL algorithms. Notably, the `learn` method can handle on-policy
    agents by calling the Gatherer.update_parameters method before gathering
    experiences.

    A ReplayBuffer is used to store and sample experiences for learning. This
    means the Learner's `agent` only interacts with the ReplayBuffer. To handle
    on-policy agents, the ReplayBuffer should be cleared at the beginning of
    the `learn` method.
    """
    @abstractmethod
    def __init__(
        self,
        agent: Agent,
        env: Env,
        num_gatherers: int,
        gather_steps_per_iteration: int,
        train_episodes_per_iteration: int,
        minibatch_size: int,
    ) -> None:
        """
        Initialize the Learner.

        This is an abstract method as the implementation of the Gatherer must
        be set depending on the RL algorithm.

        Args:
            agent (Agent): An RL agent which has an `agent_forward` method and
                a `get_action` method.
            
            env (Env): An environment for the agent to interact with. This must
                adhere to the gymnasium.Env interface.
            
            num_gatherers (int): The number of parallel Gatherers to use.

            gather_steps_per_iteration (int): The number of steps to gather
                from all Gatherers in a single iteration.
            
            train_episodes_per_iteration (int): The number of episodes/epochs
                to use for training during a single iteration.
            
            minibatch_size (int): The number of samples in a minibatch.
        
        Example:
            self.agent = agent
            self.buffer = ReplayBuffer(gather_steps_per_iteration)
            self.gatherers = [
                Gatherer.remote(agent, env)
                for i in range(num_gatherers)
            ]
        """
    
    @abstractmethod
    def learn(self) -> None:
        """
        Run one iteration of learning with parallel gatherers.

        This method adheres to the following pattern:
        1. Update gatherers with the current agent parameters.
        2. Gather experience from each gatherer in parallel.
        3. Update the agent with the gathered experience.
    
        Here is an example implementation outline:
        # Update gatherers for online experience gathering
        for gatherer in self.gatheres:
            gatherer.update_parameters.remote(self.agent.state_dict())

        # Gather experience
        results = []
        for gatherer in self.gatherers:
            results.append(gatherer.gather.remote())
        futures = ray.get(results)
        futures = [item for sublist in futures for item in sublist]
        self.buffer.add_trajectory(futures[:self.gather_steps_per_iteration])

        # Learn from gathered experience
        for step in
        """
    
    def save(self, path: Optional[str]) -> dict[str, Tensor]:
        """
        Get the current state. Optionally save it to a file.

        Args:
            path (str): The path to where the agent state will be saved.
        """
        state = self.agent.state_dict()
        if path:
            save(state, path)
        return state