from typing import Any
from typing import Optional

from gymnasium import Env

from numpy import stack

from random import random
from random import choice

from torch import Tensor

from parrl.agents.dqn_agent import DQNAgent
from parrl.core.gatherer import Gatherer
from parrl.utils.tensor_ops import split_complex_matrix


class DQNGatherer(Gatherer):

    def __init__(
        self,
        agent: DQNAgent,
        env: Env,
        steps_per_iteration: int,
        steps_per_episode: Optional[int] = None,
        epsilon: Optional[float] = 0.0,
        her_count: Optional[int] = 8,
    ) -> None:
        super().__init__(agent, env, steps_per_iteration, steps_per_episode)
        assert isinstance(agent, DQNAgent)
        self.epsilon = epsilon
        self.her_count = her_count
    
    def set_epsilon_for_exploration(self, epsilon: float) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            m = f'Epsilon must be between 0.0 and 1.0 (got {epsilon}).'
            raise ValueError(m)
        self.epsilon = epsilon
    
    def gather(self) -> dict[str, Any]:
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # For HER
        her_states = []
        her_actions = []
        her_rewards = []
        her_next_states = []
        her_dones = []

        # Statistics
        ep_lens = []
        ep_returns = []
        ep_qvalues = []
        ep_avg_qvalues = []

        episode_step = 0
        self._agent.eval()
        state, _ = self.env.reset()
        for step in range(self.steps_per_iteration):
            # Query current policy to get actions
            s = split_complex_matrix(state)
            s = Tensor(s).cpu()

            # Epsilon-greedy action selection
            q_values = self._agent.critic(s)
            if self.epsilon is not None and random() < self.epsilon:
                ac = choice(range(self.env.action_space.n))
            else:
                ac = self._agent.q_to_action(q_values)
                ac = ac.cpu().item()

            if q_values.ndim == 1:
                qval = q_values[ac].item()
            else:
                qval = q_values[0, ac].item()

            # Take environment step
            next_state, reward, done, trunc, _ = self.env.step(ac)

            states.append(state)
            actions.append(ac)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            ep_qvalues.append(qval)

            episode_step += 1
            state = next_state

            iter_end = step >= (self.steps_per_iteration - 1)
            terminate = (
                episode_step >= self.steps_per_episode
                or iter_end or done or trunc
            )

            if terminate:
                # log episode rewards and length
                start_step = step - sum(ep_lens)
                ep_lens.append(episode_step)
                ep_ret = sum(rewards[start_step:step])
                ep_returns.append(ep_ret)
                avg_qval = sum(ep_qvalues) / len(ep_qvalues)
                ep_avg_qvalues.append(avg_qval)

                # # Custom logic for Hindesign Experience Replay
                # her_transitions = []
                # for i in range(episode_step - 1):
                #     transitions = self.env.hindsight_experience_replay(
                #         i, self.her_count
                #     )
                #     her_transitions.extend(transitions)

                # for s, a, r, ns, d in her_transitions:
                #     s = stack([s.real, s.imag], axis=0)
                #     ns = stack([ns.real, ns.imag], axis=0)
                #     her_states.append(s)
                #     her_actions.append(a)
                #     her_rewards.append(r)
                #     her_next_states.append(ns)
                #     her_dones.append(d)

                # reset
                state, _ = self.env.reset()
                episode_step = 0

            if iter_end:
                # Log end of epoch information
                num_episodes = len(ep_returns)
                avg_ep_len = sum(ep_lens) / num_episodes
                avg_ep_return = sum(ep_returns) / num_episodes
                avg_ep_qvalues = sum(ep_avg_qvalues) / num_episodes

        states = states + her_states
        actions = actions + her_actions
        rewards = rewards + her_rewards
        next_states = next_states + her_next_states
        dones = dones + her_dones
        return {
            'data': {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
            },
            'stats': {
                'avg_qvalue': avg_ep_qvalues,
                'avg_ep_return': avg_ep_return,
                'avg_ep_len': avg_ep_len,
            }
        }
