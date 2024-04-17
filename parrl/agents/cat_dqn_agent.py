from torch import argmax
from torch import gather
from torch import nn
from torch import no_grad
from torch import Tensor
from torch import where
from torch import linspace

from copy import deepcopy

from einops import rearrange

from parrl.core.agent import Agent


class DQNAgent(Agent):
    """
    A DQN agent that uses HL Gauss loss instead of MSE.
    https://arxiv.org/pdf/2403.03950.pdf
    """
    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_outputs: int,
        num_categories: int,
        v_min: float,
        v_max: float,
        discount: float,
    ) -> None:
        super().__init__(discount)

        # Model architecture
        self.latent_dim = latent_dim

        class _Critic(nn.Module):
            def __init__(self, encoder: nn.Module) -> None:
                super().__init__()
                self.encoder = encoder
                num_output_logits = num_categories * num_outputs
                self.logits_head = nn.Linear(latent_dim, num_output_logits)
                self.support = linspace(v_min, v_max, num_categories)
            
            def q_values(self, x: Tensor) -> Tensor:
                logits = self(x)
                q_vals = self.logits_to_q_values(logits)
                return q_vals
            
            def logits_to_q_values(self, logits: Tensor) -> Tensor:
                # Q values are means of categorical distributions
                probs = logits.softmax(dim=-1)
                centers = (self.support[:-1] + self.support[1:]) / 2
                means = (probs * centers).sum(dim=-1)
                return means

            def forward(self, x: Tensor) -> Tensor:
                z = self.encoder(x)
                logits = self.logits_head(z)
                logits = rearrange(logits, 'n (o c) -> n o c', c=num_categories)
                return logits

        self.critic = _Critic(encoder)
        self.target = deepcopy(self.critic)
    
    def device(self) -> str:
        return self.critic.logits_head.weight.device
    
    @no_grad
    def get_action(self, x: Tensor) -> Tensor:
        q_values = self.critic.q_values(x)
        action = self.q_to_action(q_values)
        return action
    
    def critic_parameters(self) -> list[Tensor]:
        critic_params = list(self.critic.parameters())
        return critic_params
    
    def q_to_action(self, q_values: Tensor) -> Tensor:
        next_gates = argmax(q_values, dim=-1)
        return next_gates

    def target_value(
        self,
        next_state: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        """
        Compute the target values. The critic network is used
        here to do double_q learning.
        """
        with no_grad():
            # Target computes values
            next_qa_values = self.target.q_values(next_state)
            # Network other than target selects values
            double_q_critic = self.critic.q_values(next_state)
            next_action = self.q_to_action(double_q_critic)
            next_action = next_action.unsqueeze(dim=-1).long()

            next_q_values = gather(
                next_qa_values, dim=1, index=next_action,
            )
            next_q_values = next_q_values.squeeze()
            reward = reward.float()
            target_values = reward + (1 - done) * self.discount * next_q_values
            target_values = target_values.squeeze()
        return target_values

    def q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute the vector of q_values for all actions.
        """
        qa_values = self.critic.q_values(state)
        q_indices = action.unsqueeze(dim=-1).long()
        q_values = gather(qa_values, dim=1, index=q_indices)
        q_values = q_values.squeeze()
        return q_values

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> tuple[Tensor, Tensor]:
        logits = self.critic(state)[action]
        target_values = self.target_value(next_state, reward, done)
        return logits, target_values