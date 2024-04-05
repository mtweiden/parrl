from gymnasium import Env
from gymnasium import make

import ray

from parrl.agents.gauss_dqn_agent import GaussDQNAgent
from parrl.gatherers.gauss_dqn_gatherer import GaussDQNGatherer
from parrl.networks.mlp_encoder import MLPEncoder


def simple_env() -> Env:
    return make('CartPole-v1')


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

class TestDQNGatherer:

    def test_setup(self) -> None:
        simple_env()
        gauss_agent()

    def test_constructor(self) -> None:
        env = simple_env()
        agent = gauss_agent()
        gatherer = GaussDQNGatherer(agent, env, 100)
        assert gatherer.get_agent() == agent
        assert gatherer.env == env
        assert gatherer.steps_per_iteration == 100
    
    def test_remote(self) -> None:
        env = simple_env()
        agent = gauss_agent()
        remote_gatherer = ray.remote(num_cpus=1)(GaussDQNGatherer)
        remote_gatherer = remote_gatherer.remote(agent, env, 100)
        remote_agent = ray.get(remote_gatherer.get_agent.remote())
        local_params = agent.critic_parameters()
        remote_params = remote_agent.critic_parameters()
        assert all((l == r).all() for l, r in zip(local_params, remote_params))

    def test_gather(self) -> None:
        env = simple_env()
        agent = gauss_agent()
        gatherer = GaussDQNGatherer(agent, env, 100, 25)
        data = gatherer.gather()
        assert 'stats' in data
        assert 'data' in data
