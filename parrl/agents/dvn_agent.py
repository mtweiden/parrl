from __future__ import annotations

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
from parrl.core.model import Model
from parrl.networks.softmoe import SoftMoELayer
from parrl.networks.noisy import NoisyLinear


class DVNAgent(Agent):
    """
    A Model Value Network acts like a DQN, but only learns the value of states.

    Q values are determined by modeling state transitions then taking the value
    of those states. This is feasible when modeling state transitions can be
    done with very high accuracy.

    TODO(Mathias): Rewrite to handle dynamic actions.
    """
    def __init__(
        self,
        model: Model,
        encoder: nn.Module,
        latent_dim: int,
        discount: float,
        noisy_net: bool = False,
        num_experts: int = 0,
        expert_latent_dim: int = 0,
    ) -> None:
        super().__init__(discount)
        self.model = model

        # Architecture parameters
        self.latent_dim = latent_dim
        self.expert_latent_dim = expert_latent_dim
        self.noisy_net = noisy_net

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
                output_size = (latent_dim, 1)
                if noisy_net:
                    linear_out = NoisyLinear(*output_size)
                else:
                    linear_out = nn.Linear(*output_size)
                self.val_head = nn.Sequential(
                    nn.GELU(),
                    linear_out,
                )

            def forward(self, x: Tensor) -> Tensor:
                z = self.encoder(x)
                if self.do_softmoe:
                    z = rearrange(z, 'b (m d) -> b m d', d=expert_latent_dim)
                    z = self.softmoe(z)
                    z = rearrange(z, 'b m d -> b (m d)')
                value = self.val_head(z)
                return value

        self.critic = _Critic(encoder)
        self.target = deepcopy(self.critic)

    def device(self) -> device:
        return self.critic.val_head[1].weight.device
    
    def critic_parameters(self) -> list[Parameter]:
        critic_params = list(self.critic.parameters())
        return critic_params

    def model_q_values(self, value_network: nn.Module, state: Tensor) -> Tensor:
        # next_states: (bs, num_actions, state_dim)
        next_states = self.model(state)
        q_values = value_network(next_states)
        return q_values
    
    @no_grad  # type: ignore
    def get_action(self, x: Tensor) -> Tensor:
        q_values = self.model_q_values(self.critic, x)
        action = self.q_to_action(q_values)
        return action
    
    def q_to_action(self, q_values: Tensor) -> Tensor:
        next_gates = argmax(q_values, dim=-1)
        return next_gates

    @no_grad  # type: ignore
    def target_value(
        self,
        next_state: Tensor,
        reward: Tensor,
        done: Tensor,
    ) -> Tensor:
        """
        Compute the target values. The critic network is used here to do 
        double_q learning.

        TODO(Mathias): Generalize so that triple_q or quadruple_q learning can
        be done. This can be done if the done and reward functions are deter-
        ministic functions of only the state.

        single_q_critic = self.model_q_values(self.critic, state)
        single_q_critic = rearrange(single_q_critic, 'a n ... -> n a ...')
        action = self.q_to_action(single_q_critic).int()
        next_state = self.model.take_action(state, action)
        """
        # Network other than target selects values
        double_q_critic = self.model_q_values(self.critic, next_state)
        double_q_critic = rearrange(double_q_critic, 'a n ... -> n a ...')
        next_action = self.q_to_action(double_q_critic).int()
        next_next_state = self.model.take_action(next_state, next_action)

        # Target computes values
        next_q_value = self.target(next_next_state)

        reward = reward.float()
        target_value = reward + (1 - done) * self.discount * next_q_value
        return target_value

    def q_value(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Compute the vector of q_values for all actions.
        """
        qa_values = self.model_q_values(self.critic, state)
        q_indices = action.unsqueeze(dim=-1).long()
        q_values = gather(qa_values, dim=1, index=q_indices)
        q_values = q_values.squeeze()
        return q_values

    def reset_noise(self) -> None:
        if self.noisy_net:
            self.critic.adv_head[1].reset_noise()
            self.target.adv_head[1].reset_noise()

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
