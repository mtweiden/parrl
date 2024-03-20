from gymnasium import Env
from gymnasium import make

import ray

from parrl.agents.ppo_agent import PPOAgent
from parrl.gatherers.ppo_gatherer import PPOGatherer
from parrl.networks.mlp_encoder import MLPEncoder


def simple_env() -> Env:
    return make('CartPole-v1')


def simple_agent() -> PPOAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = PPOAgent(encoder, 64, num_outputs, 0.99, 0.95)
    return agent


class TestPPOGatherer:

    def test_setup(self) -> None:
        simple_env()
        simple_agent()

    def test_constructor(self) -> None:
        env = simple_env()
        agent = simple_agent()
        gatherer = PPOGatherer(agent, env, 100)
        assert gatherer.get_agent() == agent
        assert gatherer.env == env
        assert gatherer.steps_per_iteration == 100
    
    def test_remote(self) -> None:
        env = simple_env()
        agent = simple_agent()
        remote_gatherer = ray.remote(num_cpus=1)(PPOGatherer)
        remote_gatherer = remote_gatherer.remote(agent, env, 100)
        remote_agent = ray.get(remote_gatherer.get_agent.remote())
        local_params = agent.actor_parameters()
        remote_params = remote_agent.actor_parameters()
        assert all((l == r).all() for l, r in zip(local_params, remote_params))

    def test_gather(self) -> None:
        env = simple_env()
        agent = simple_agent()
        gatherer = PPOGatherer(agent, env, 100, 25)
        data = gatherer.gather()
        assert 'stats' in data
        assert 'data' in data