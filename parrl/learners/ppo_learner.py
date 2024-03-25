from typing import Any
from typing import Optional
from typing import Sequence

from gymnasium import Env

from numpy import ceil

import ray

from torch import clamp
from torch import exp
from torch import min
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from parrl.core.buffer import ReplayBuffer
from parrl.core.buffer import ReplayBufferDataset
from parrl.core.learner import Learner

from parrl.agents.ppo_agent import PPOAgent
from parrl.gatherers.ppo_gatherer import PPOGatherer


class PPOLearner(Learner):
    """A Learner for on-policy PPO agents."""
    def __init__(
        self,
        agent: PPOAgent,
        env: Env,
        num_gatherers: int,
        gather_steps_per_iteration: int,
        train_episodes_per_iteration: int,
        minibatch_size: int,
        gradient_clip: float,
        learning_rate: float | Sequence[float],
        entropy_bonus: float,
        ppo_clip: float,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the Learner.

        This is an abstract method as the implementation of the Gatherer must
        be set depending on the RL algorithm.

        Args:
            agent (Agent): An RL agent which has a `forward` method and a
                `get_action` method.
            
            env (Env): An environment for the agent to interact with. This must
                adhere to the gymnasium.Env interface.
            
            num_gatherers (int): The number of parallel Gatherers to use.

            gather_steps_per_iteration (int): The number of steps to gather
                from all Gatherers in a single iteration.
            
            train_episodes_per_iteration (int): The number of episodes/epochs
                to use for training during a single iteration.
            
            minibatch_size (int): The number of samples in a minibatch.

            gradient_clip (float): The maximum value for the gradient norm.

            learning_rate (float | Sequence[float]): The learning rate for the
                agent's optimizer. If a Sequence is given the first rate is for
                the actor and the second is for the critic.

            entropy_bonus (float): The entropy bonus multiplier hyperparameter
                for the agent's policy.

            ppo_clip (float): The clipping value for the PPO surrogate loss.
        """
        self.agent = agent
        self.buffer = ReplayBuffer(gather_steps_per_iteration)
        self.gather_steps_per_iteration = gather_steps_per_iteration
        self.train_episodes_per_iteration = train_episodes_per_iteration
        self.minibatch_size = minibatch_size
        self.gradient_clip = gradient_clip
        self.entropy_bonus = entropy_bonus
        self.ppo_clip = ppo_clip

        self.steps_per_gatherer = int(
            ceil(gather_steps_per_iteration / num_gatherers)
        )
        remote_gatherer = ray.remote(num_cpus=1)(PPOGatherer)
        self.gatherers = [
            remote_gatherer.remote(agent, env, self.steps_per_gatherer)
            for _ in range(num_gatherers)
        ]

        if isinstance(learning_rate, float):
            self.learning_rates = (learning_rate,)
        else:
            self.learning_rates = tuple(learning_rate)
        self.optimizers = {
            'actor': AdamW(
                self.agent.actor_parameters(),
                lr=self.learning_rates[0],
            ),
            'critic': AdamW(
                self.agent.critic_parameters(),
                lr=self.learning_rates[1 % len(self.learning_rates)],
            ),
        }

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
        2. Gather (on-policy) experience from each gatherer in parallel.
        3. Update the agent with the gathered experience which is stored in the
           ReplayBuffer for `train_episodes_per_iteration` number of episodes.
        
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
        for episode in range(self.train_episodes_per_iteration):
            dataloader = self._prepare_dataloader()
            for batch in dataloader:
                actor_loss, critic_loss = self._train_step(batch)
                if self.do_logging:
                    wandb.log(
                        {'actor_loss': actor_loss, 'critic_loss': critic_loss}
                    )
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
            flat_data.extend(self.buffer.group_data(d))
        self.buffer.clear()
        self.buffer.add_experience(flat_data)
    
    def _prepare_dataloader(self) -> DataLoader:
        dataset = ReplayBufferDataset(self.buffer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    def _train_step(self, batch: Tensor) -> tuple[float, float]:
        """
        Train the agent using the gathered experience.

        This method is called by the `learn` method and is responsible for
        training the agent using the gathered experience.

        Args:
            batch (Tensor): A minibatch of experience from the ReplayBuffer.
        
        Returns:
            actor_loss (Tensor): The loss of the actor.

            critic_loss (Tensor): The loss of the critic.
        """
        batch = tuple(t.to(self.agent.device()).float() for t in batch)
        s, ac, logp, tarv, adv, ent = batch

        self.optimizers['actor'].zero_grad()
        logits = self.agent.actor_forward(s)
        actor_loss = self._actor_loss(logits, ac, logp, adv, ent)
        actor_loss.backward()
        clip_grad_norm_(self.agent.actor_parameters(), self.gradient_clip)
        self.optimizers['actor'].step()

        self.optimizers['critic'].zero_grad()
        value = self.agent.critic_forward(s)
        critic_loss = self._critic_loss(value, tarv)
        critic_loss.backward()
        clip_grad_norm_(self.agent.critic_parameters(), self.gradient_clip)
        self.optimizers['critic'].step()

        return actor_loss.item(), critic_loss.item()
    
    def _format_stats(
        self,
        stats: list[dict[str, float | Sequence[float]]],
    ) -> None:
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
                    new_stats = list(gatherer_stats[stat_name])
                    combined_stats[stat_name].extend(new_stats)
                else:
                    combined_stats[stat_name].append(gatherer_stats[stat_name])
        combined_stats = {k: sum(v) / len(v) for k, v in combined_stats.items()}
        combined_stats['iteration'] = self.iteration
        return combined_stats
    
    def _actor_loss(
        self,
        logits: Tensor,
        action: Tensor,
        old_logp: Tensor,
        advantages: Tensor,
        entropies: Tensor,
    ) -> Tensor:
        """
        Compute the clipped PPO surrogate loss.

        Args:
            logits (Tensor): The current policy's logits.

            state (Tensor): The state from which the action was taken. This
                is sampled from the ReplayBuffer.

            action (Tensor): The action taken. This is sampled from the
                ReplayBuffer.

            old_logp (Tensor): The log probability of the action under the
                previous policy. This is sampled from the ReplayBuffer.

            advantages (Tensor): The advantages of the action. This is sampled
                from the ReplayBuffer.

            entropies (Tensor): The entropy of the action distribution. This is
                sampled from the ReplayBuffer.
        """
        ac_dist = Categorical(logits=logits)
        logp = ac_dist.log_prob(action)
        # important weighting
        ratio = exp(logp - old_logp)
        clip_ratio = clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip)
        unclipped_adv = ratio * advantages
        clipped_adv = clip_ratio * advantages
        loss = -1 * min(unclipped_adv, clipped_adv)
        loss = loss - self.entropy_bonus * entropies
        loss = loss.mean()
        return loss

    def _critic_loss(self, value: Tensor, returns: Tensor) -> Tensor:
        """
        The loss function for the PPOAgent's critic.

        Args:
            value (Tensor): The state values predicted by the critic.

            returns (Tensor): The actual returns from the environment.
        """
        loss = pow((returns - value), 2).mean()
        return loss