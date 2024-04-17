from gymnasium import Env
from gymnasium import make

from torch import tensor

from parrl.agents.dqn_agent import DQNAgent
from parrl.agents.gauss_dqn_agent import GaussDQNAgent
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


def gauss_agent() -> GaussDQNAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]  # type: ignore
    num_outputs = env.action_space.n  # type: ignore
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = GaussDQNAgent(
        encoder,
        64,
        num_outputs,
        0.99,
        num_experts=4,
        expert_latent_dim=16,
        v_min=-1.0,
        v_max=200.0,
        num_bins=51,
    )
    return agent


def noisy_agent() -> DQNAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]  # type: ignore
    num_outputs = env.action_space.n  # type: ignore
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = DQNAgent(encoder, 64, num_outputs, 0.99, noisy_net=True)
    return agent


class TestDQNAgent:

    def test_setup(self) -> None:
        simple_env()
        simple_agent()
        moe_agent()
        gauss_agent()
        noisy_agent()

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

    def test_gauss_critic(self) -> None:
        env = simple_env()
        agent = gauss_agent()
        s, _ = env.reset()
        s = tensor(s).unsqueeze(0)
        qa_logits = agent.critic(s)
        assert qa_logits.shape[-2] == env.action_space.n  # type: ignore
        assert qa_logits.shape[-1] == agent.num_bins
        qa_vals = agent.from_logits(qa_logits)
        assert qa_vals.shape[-1] == env.action_space.n  # type: ignore
