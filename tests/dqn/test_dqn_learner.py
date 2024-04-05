from gymnasium import Env
from gymnasium import make

import ray

from torch import tensor

from parrl.agents.dqn_agent import DQNAgent
from parrl.networks.mlp_encoder import MLPEncoder
from parrl.learners.dqn_learner import DQNLearner


def simple_env() -> Env:
    return make('CartPole-v1')


def simple_agent() -> DQNAgent:
    env = simple_env()
    input_dim = env.observation_space.shape[0]
    num_outputs = env.action_space.n
    dims = [64, 64]
    encoder = MLPEncoder(input_dim, dims)
    agent = DQNAgent(encoder, 64, num_outputs, 0.99)
    return agent


def default_kwargs() -> dict:
    return {
        'num_gatherers': 4,
        'gather_steps_per_iteration': 500,
        'train_steps_per_iteration': 10,
        'minibatch_size': 32,
        'gradient_clip': 0.5,
        'learning_rate': 1e-3,
        'target_update_period': 10000,
        'buffer_size': 1_000_000,
        'exploration_epsilon': 0.1,
    }

class TestDQNLearner:

    def test_setup(self) -> None:
        simple_env()
        simple_agent()

    def test_constructor(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = DQNLearner(agent, env, **kwargs)
        assert learner.agent == agent
   
    def test_update_parameters(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = DQNLearner(agent, env, **kwargs)
        state_dict = learner.agent.state_dict()
        new_state_dict = {k: -1*v for k, v in state_dict.items()}
        learner.agent.load_state_dict(new_state_dict)
        learner._update_remote_parameters()
        for gatherer in learner.gatherers:
            remote_state_dict = ray.get(gatherer.get_agent.remote()).state_dict()
            assert all(
                (new_state_dict[k] == remote_state_dict[k]).all()
                for k in new_state_dict
            )

    def test_train_step(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = DQNLearner(agent, env, **kwargs)
        data = ray.get(learner.gatherers[0].gather.remote())

        states = data['data']['states']
        actions = data['data']['actions']
        rewards = data['data']['rewards']
        nstates = data['data']['next_states']
        dones = data['data']['dones']

        # Add to replay buffer
        for s, a, r, ns, d in zip(states, actions, rewards, nstates, dones):
            learner.buffer.store(s, a, r, ns, d)

        s, a, r, ns, d, w, i = learner.buffer.sample_batch(learner.minibatch_size)
        s = tensor(s).float()
        a = tensor(a).long()
        r = tensor(r).float()
        ns = tensor(ns).float()
        d = tensor(d).float()
        w = tensor(w).float()
        i = tensor(i).long()
        batch = s, a, r, ns, d, w, i
        
        c_loss = learner._train_step(batch, 0)

        assert isinstance(c_loss, float)

    def test_prepare_dataloader(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = DQNLearner(agent, env, **kwargs)
        data = [ray.get(learner.gatherers[0].gather.remote())['data']]

        learner._prepare_buffer(data)
        dataloader = learner._prepare_dataloader()
        for batch in dataloader:
            assert len(batch) == 7
            assert len(batch[0]) == kwargs['minibatch_size']

    def test_learn(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = DQNLearner(agent, env, **kwargs)
        learner.learn()
