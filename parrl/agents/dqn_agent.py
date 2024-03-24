from torch import argmax
from torch import gather
from torch import nn
from torch import no_grad
from torch import Tensor
from torch import where
from torch.distributions import Categorical

from copy import deepcopy

from parrl.core.agent import Agent


class DQNAgent(Agent):

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_outputs: int,
        discount: float,
    ) -> None:
        super().__init__(discount)

        # Model architecture
        self.latent_dim = latent_dim
        self.critic = nn.Sequential(encoder, nn.Linear(latent_dim, num_outputs))
        self.target = deepcopy(self.critic)
    
    def device(self) -> str:
        return self.critic[-1].weight.device
    
    @no_grad
    def get_action(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)
        logits = self.critic(x)
        distribution = Categorical(logits=logits)
        sample = distribution.sample()
        return sample
    
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
            next_qa_values = self.target(next_state)
            # Network other than target selects values
            double_q_critic = self.critic(next_state)
            next_action = self.q_to_action(double_q_critic)
            next_action = next_action.unsqueeze(dim=-1).long()

            next_q_values = gather(
                next_qa_values, dim=1, index=next_action,
            )
            next_q_values = next_q_values.squeeze()
            reward = reward.float()
            target_values = where(
                done, reward, reward + self.discount * next_q_values
            )
            target_values = target_values.squeeze()
        return target_values

    def q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute the vector of q_values for all actions.
        """
        qa_values = self.critic(state)
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
        q_values = self.q_value(state, action)
        target_values = self.target_value(next_state, reward, done)
        return q_values, target_values