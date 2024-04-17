from typing import Sequence

from parrl.agents.dqn_agent import DQNAgent
from parrl.networks.unitary_encoder import UnitaryEncoder


class UnitaryDQNAgent(DQNAgent):

    def __init__(
        self,
        num_qubits: int,
        layer_dims: Sequence[int],
        nerf_dim: int,
        dropout: float,
        num_outputs: int,
        discount: float,
        noisy_net: bool = False,
        num_experts: int = 0,
        expert_latent_dim: int = 0,
    ) -> None:
        encoder = UnitaryEncoder(
            num_qubits=num_qubits,
            layer_dims=layer_dims,
            nerf_dim=nerf_dim,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=layer_dims[-1],
            num_outputs=num_outputs,
            discount=discount,
            noisy_net=noisy_net,
            num_experts=num_experts,
            expert_latent_dim=expert_latent_dim,
        )
