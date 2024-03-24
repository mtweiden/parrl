from typing import Any

from torch import nn
from torch import no_grad
from torch import Tensor
from torch.distributions import Categorical

from parrl.core.agent import Agent


class PPOAgent(Agent):

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        num_outputs: int,
        discount: float,
        gae_discount: float,
    ) -> None:
        super().__init__(discount)

        # Model architecture
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.logit_head = nn.Sequential(
            nn.Linear(latent_dim, num_outputs),
        )
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # PPO hyperparameters
        self.gae_discount = gae_discount
    
    def device(self) -> str:
        return self.logit_head[0].weight.device
    
    def actor_forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        logits = self.logit_head(z)
        return logits
    
    def critic_forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        value = self.value_head(z)
        return value

    def agent_forward(self, x: Tensor) -> Any:
        z = self.encoder(x)
        logits = self.logit_head(z)
        value = self.value_head(z)
        return logits, value
    
    @no_grad
    def get_action(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(dim=0)
        logits = self.logit_head(x)
        distribution = Categorical(logits=logits)
        sample = distribution.sample()
        return sample
    
    def actor_parameters(self) -> list[Tensor]:
        encoder_params = list(self.encoder.parameters())
        actor_params = list(self.logit_head.parameters())
        return encoder_params + actor_params
    
    def critic_parameters(self) -> list[Tensor]:
        critic_params = list(self.value_head.parameters())
        return critic_params