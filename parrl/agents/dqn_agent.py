from einops import rearrange

from torch import argmax
from torch import device
from torch import gather
from torch import nn
from torch import no_grad
from torch import Tensor
from torch.nn import Parameter

from copy import deepcopy

from parrl.core.agent import Agent
from parrl.networks.softmoe import SoftMoELayer


class DQNAgent(Agent):

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_outputs: int,
        discount: float,
        num_experts: int = 0,
        expert_latent_dim: int = 0,
    ) -> None:
        super().__init__(discount)

        # Model architecture
        self.latent_dim = latent_dim
        self.expert_latent_dim = expert_latent_dim

        class _Critic(nn.Module):
            def __init__(self, encoder: nn.Module) -> None:
                super().__init__()
                self.encoder = encoder
                if num_experts == 0 or expert_latent_dim == 0:
                    self.do_softmoe = False
                else:
                    assert latent_dim % expert_latent_dim == 0
                    self.do_softmoe = True
                    self.softmoe = SoftMoELayer(
                        expert_latent_dim,
                        expert_latent_dim,
                        num_experts,
                        latent_dim // num_experts,
                    )
                self.adv_head = nn.Sequential(
                    nn.GELU(),
                    nn.Linear(latent_dim, num_outputs),
                )
                self.val_head = nn.Sequential(
                    nn.GELU(),
                    nn.Linear(latent_dim, 1),
                )

            def forward(self, x: Tensor) -> Tensor:
                z = self.encoder(x)
                if self.do_softmoe:
                    z = rearrange(z, 'b (m d) -> b m d', d=expert_latent_dim)
                    z = self.softmoe(z)
                    z = rearrange(z, 'b m d -> b (m d)')
                value = self.val_head(z)
                advantages = self.adv_head(z)
                adv_mean = advantages.mean(dim=-1, keepdim=True)
                q_values = value + (advantages - adv_mean)
                return q_values

        self.critic = _Critic(encoder)
        self.target = deepcopy(self.critic)
    
    def device(self) -> device:
        return self.critic.adv_head.weight.device
    
    @no_grad
    def get_action(self, x: Tensor) -> Tensor:
        q_values = self.critic(x)
        action = self.q_to_action(q_values)
        return action
    
    def critic_parameters(self) -> list[Parameter]:
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
            target_values = reward + (1 - done) * self.discount * next_q_values
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
