from typing import Any
from typing import Optional
from typing import Sequence

from gymnasium import Env

from numpy import ceil

import ray

from torch import Tensor
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from parrl.buffers.dqn_buffer import PrioritizedReplayBuffer as ReplayBuffer
from parrl.buffers.dqn_dataset import PrioritizedReplayDataset as Dataset
from parrl.core.utils.buffer_utils import group_data
from parrl.core.learner import Learner

from parrl.agents.gauss_dvn_agent import GaussDVNAgent
from parrl.gatherers.gauss_dvn_gatherer import GaussDVNGatherer


class GaussDVNLearner(Learner):
    """A Learner for DVN agents."""
    def __init__(
        self,
        agent: GaussDVNAgent,
        env: Env,
        num_gatherers: int,
        gather_steps_per_iteration: int,
        train_steps_per_iteration: int,
        minibatch_size: int,
        gradient_clip: float,
        learning_rate: float,
        target_update_period: int,
        buffer_size: int,
        exploration_epsilon: float,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Learner.

        This is an abstract method as the implementation of the Gatherer must
        be set depending on the RL algorithm.

        Args:
            agent (Agent): An RL agent which has an `forward` method and a
                `get_action` method.
            
            env (Env): An environment for the agent to interact with. This must
                adhere to the gymnasium.Env interface.
            
            num_gatherers (int): The number of parallel Gatherers to use.

            gather_steps_per_iteration (int): The number of steps to gather
                from all Gatherers in a single iteration.
            
            train_steps_per_iteration (int): The number of steps per training
                iteration.
            
            minibatch_size (int): The number of samples in a minibatch.

            gradient_clip (float): The maximum value for the gradient norm.

            learning_rate (float): The learning rate for the agent's optimizer. 
            
            target_update_period (int): The number of steps before the target
                network is updated.
            
            buffer_size (int): The size of the ReplayBuffer.

            exploration_epsilon (float): The epsilon value for exploration.
        """
        self.agent = agent
        self.buffer = ReplayBuffer(buffer_size)
        self.gather_steps_per_iteration = gather_steps_per_iteration
        self.train_steps_per_iteration = train_steps_per_iteration
        self.minibatch_size = minibatch_size
        self.gradient_clip = gradient_clip
        self.target_update_period = target_update_period
        self.learning_rate = learning_rate

        self.steps_per_gatherer = int(
            ceil(gather_steps_per_iteration / num_gatherers)
        )
        remote_gatherer = ray.remote(num_cpus=1)(GaussDVNGatherer)
        self.gatherers = [
            remote_gatherer.remote(
                agent, env, self.steps_per_gatherer, epsilon=exploration_epsilon
            ) for _ in range(num_gatherers)
        ]

        self.optimizer = AdamW(
            self.agent.critic_parameters(),
            lr=self.learning_rate,
        )

        self.iteration = 0
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.do_logging = project_name is not None and experiment_name is not \
            None
    
    def learn(self) -> dict[str, Any]:
        """
        Run one iteration of PPO learning with parallel gatherers.

        This method adheres to the following pattern:
        1. Update gatherers with the current agent parameters.
        2. Gather experience from each gatherer in parallel.
        3. Update the agent with the gathered experience which is stored in the
           ReplayBuffer for `train_steps_per_iteration` number of steps.
        
        Returns:
            (dict[str, Any]): A dict of training statistics.
        """
        # Update gatherers for online experience gathering
        self._update_remote_parameters()
        # Gather experience
        results = []
        for gatherer in self.gatherers:
            results.append(gatherer.gather.remote())
        futures = ray.get(results)
        # Put experience into the ReplayBuffer
        stats = [f['stats'] for f in futures]
        data = [f['data'] for f in futures]
        self._prepare_buffer(data)
        # Learn from gathered experience
        self.agent = self.agent.train().to(self.device)
        step = 0
        while step < self.train_steps_per_iteration:
            dataloader = self._prepare_dataloader()
            for batch in dataloader:
                critic_loss = self._train_step(batch, step)
                step += 1
                # Handle early exits
                if step >= self.train_steps_per_iteration:
                    break
                if self.do_logging:
                    wandb.log({'critic_loss': critic_loss})
        stats = self._format_stats(stats)
        # Record statistics
        if self.do_logging:
            wandb.log(stats)
        self.iteration += 1
        return stats
        
    def _update_remote_parameters(self) -> None:
        state_dict = {k: v.cpu() for k, v in self.agent.state_dict().items()}
        for gatherer in self.gatherers:
            gatherer.update_parameters.remote(state_dict)
    
    def _prepare_buffer(self, data: list[dict[str, Tensor]]) -> None:
        flat_data = []
        for d in data:
            flat_data.extend(group_data(d))  # type: ignore
        for d in flat_data:
            self.buffer.store(*d)
    
    def _prepare_dataloader(self) -> DataLoader:
        dataset = Dataset(self.buffer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            # shuffle=True,  $ This is just extra work for PER
            drop_last=True,
        )
        return dataloader

    def _train_step(self, batch: Tensor, batch_num: int) -> float:
        """
        Train the agent using the gathered experience.

        This method is called by the `learn` method and is responsible for
        training the agent using the gathered experience.

        Args:
            batch (Tensor): A minibatch of experience from the ReplayBuffer.

            batch_num (int): The step in the current episode.
        
        Returns:
            critic_loss (Tensor): The loss of the critic.
        """
        batch = tuple(
            t.to(self.agent.device()).float() for t in batch
        )  # type: ignore
        s, a, r, ns, d, weights, indices = batch

        # Update the critic
        self.optimizer.zero_grad()
        q_logits, target_values = self.agent(s, a, r, ns, d)
        full_critic_loss = self.critic_loss(q_logits, target_values, weights)
        critic_loss = full_critic_loss.mean()
        critic_loss.backward()
        clip_grad_norm_(self.agent.critic_parameters(), self.gradient_clip)
        self.optimizer.step()

        # Update the target network
        if batch_num % self.target_update_period == 0:
            self.agent.target.load_state_dict(self.agent.critic.state_dict())

        # Update priorities in the ReplayBuffer
        new_priorities = full_critic_loss.abs().detach().cpu().numpy() + 1e-6
        indices = indices.int().tolist()
        self.buffer.update_priorities(indices, new_priorities)

        return critic_loss.item()

    def critic_loss(
        self,
        q_logits: Tensor,
        target_values: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        loss = self.agent.hlgauss(q_logits, target_values)
        if weights is not None:
            assert weights.shape == loss.shape
            loss = loss * weights
        return loss
    
    def _format_stats(
        self,
        stats: list[dict[str, float | Sequence[float]]],
    ) -> dict[str, Any]:
        """
        Format statistics for logging.

        Args:
            stats (list[dict[str, float | Sequence[float]]]): Statistics from
                the gatherers.
        """
        # Unpack stats
        combined_stats = {}
        for gatherer_stats in stats:
            for stat_name in gatherer_stats:
                if stat_name not in combined_stats:
                    combined_stats[stat_name] = []
                if isinstance(gatherer_stats[stat_name], (list, tuple)):
                    new_stats = list(gatherer_stats[stat_name])  # type: ignore
                    combined_stats[stat_name].extend(new_stats)
                else:
                    combined_stats[stat_name].append(gatherer_stats[stat_name])
        combined_stats = {k: sum(v) / len(v) for k, v in combined_stats.items()}
        combined_stats['iteration'] = self.iteration
        return combined_stats
