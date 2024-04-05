from gymnasium import Env
from gymnasium import make

from torch import tensor

from parrl.agents.dqn_agent import DQNAgent
from parrl.networks.mlp_encoder import MLPEncoder


def simple_env() -> Env:
    return make('CartPole-v1')


def simple_agent() -> DQNAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]  # type: ignore
    num_outputs = env.action_space.n  # type: ignore
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = DQNAgent(encoder, 64, num_outputs, 0.99)
    return agent


def moe_agent() -> DQNAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]  # type: ignore
    num_outputs = env.action_space.n  # type: ignore
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = DQNAgent(
        encoder,
        64,
        num_outputs,
        0.99,
        num_experts=4,
        expert_latent_dim=16,
    )
    return agent


class TestDQNAgent:

    def test_setup(self) -> None:
        simple_env()
        simple_agent()
        moe_agent()

    def test_simple_critic(self) -> None:
        env = simple_env()
        agent = simple_agent()
        s, _ = env.reset()
        s = tensor(s).unsqueeze(0)
        qa_vals = agent.critic(s)
        assert qa_vals.shape[-1] == env.action_space.n  # type: ignore

    def test_moe_critic(self) -> None:
        env = simple_env()
        agent = moe_agent()
        s, _ = env.reset()
        s = tensor(s).unsqueeze(0)
        qa_vals = agent.critic(s)
        assert qa_vals.shape[-1] == env.action_space.n  # type: ignore
