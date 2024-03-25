from gymnasium import Env
from gymnasium import make

import ray

from torch import tensor

from parrl.agents.ppo_agent import PPOAgent
from parrl.networks.mlp_encoder import MLPEncoder
from parrl.learners.ppo_learner import PPOLearner


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


def default_kwargs() -> dict:
    return {
        'num_gatherers': 4,
        'gather_steps_per_iteration': 500,
        'train_episodes_per_iteration': 10,
        'minibatch_size': 32,
        'gradient_clip': 0.5,
        'learning_rate': 1e-3,
        'entropy_bonus': 0.01,
        'ppo_clip': 0.2,
    }

class TestPPOLearner:

    def test_setup(self) -> None:
        simple_env()
        simple_agent()

    def test_constructor(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = PPOLearner(agent, env, **kwargs)
        assert learner.agent == agent
   
    def test_update_parameters(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = PPOLearner(agent, env, **kwargs)
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
        learner = PPOLearner(agent, env, **kwargs)
        data = ray.get(learner.gatherers[0].gather.remote())

        s = tensor(data['data']['states'])
        ac = tensor(data['data']['actions'])
        logp = tensor(data['data']['logps'])
        tvars = tensor(data['data']['target_values'])
        advs = tensor(data['data']['advantages'])
        ents = tensor(data['data']['entropies'])

        batch = (s, ac, logp, tvars, advs, ents)
        a_loss, c_loss = learner._train_step(batch)

        assert isinstance(a_loss, float)
        assert isinstance(c_loss, float)

    def test_prepare_dataloader(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = PPOLearner(agent, env, **kwargs)
        data = [ray.get(learner.gatherers[0].gather.remote())['data']]

        learner._prepare_buffer(data)
        dataloader = learner._prepare_dataloader()
        for batch in dataloader:
            assert len(batch) == 6
            assert len(batch[0]) == kwargs['minibatch_size']

    def test_learn(self) -> None:
        env = simple_env()
        agent = simple_agent()
        kwargs = default_kwargs()
        learner = PPOLearner(agent, env, **kwargs)
        learner.learn()