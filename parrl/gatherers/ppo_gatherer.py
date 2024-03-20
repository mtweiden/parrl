from typing import Any
from typing import Optional
from typing import Sequence

from gymnasium import Env

from torch import tensor
from torch.distributions import Categorical

from parrl.agents.ppo_agent import PPOAgent
from parrl.core.gatherer import Gatherer
from parrl.utils.tensor_ops import split_complex_matrix


class PPOGatherer(Gatherer):

    def __init__(
        self,
        agent: PPOAgent,
        env: Env,
        steps_per_iteration: int,
        steps_per_episode: Optional[int] = None,
    ) -> None:
        super().__init__(agent, env, steps_per_iteration, steps_per_episode)
        assert isinstance(agent, PPOAgent)

    def compute_return(
        self,
        rewards: Sequence[float],
        discount: Optional[float] = None,
    ) -> list[float]:
        returns_r = []
        return_to_go = 0
        discount = self._agent.discount if discount is None else discount
        for rew in reversed(rewards):
            return_to_go = (self._agent.discount * return_to_go) + rew
            returns_r.append(return_to_go)
        return [r for r in reversed(returns_r)]

    def compute_advantage(
        self,
        rewards: Sequence[float],
        critic_values: Sequence[float],
    ) -> list[float]:
        # GAE computation for PPO2
        delta = [
            rewards[i] + self._agent.discount * critic_values[i+1] \
                - critic_values[i]
            for i in range(len(rewards) - 1)
        ]
        discount = self._agent.discount * self._agent.gae_discount
        advantages = self.compute_return(delta, discount)
        return advantages

    def gather(self) -> dict[str, Any]:
        """
        Gather experience from the environment using the Gatherer's agent.

        Returns:
            (list): A list of concatenated state transitions. These can be
                from the same trajectory or from different trajectories,
                depending on the implementation.
        """
        # The following lists are required for training
        states = []
        actions = []
        logps = []
        target_values = []
        advantages = []
        entropies = []

        # ep_values measure agent's perceived value of states
        ep_values = []
        # ep_rewards measure the actual rewards received
        ep_rewards = []
        # ep_entropies are used to measure average entropy
        ep_entropies = []

        # Stats
        avg_entropies = []
        avg_advantages = []
        avg_rewards = []

        episode_step = 0

        self._agent.eval()
        state, _ = self.env.reset()
        for step in range(self.steps_per_iteration):
            # Query current policy
            s = split_complex_matrix(state)
            s = tensor(s).cpu()

            logits, critic_value = self._agent(s)
            action_dist = Categorical(logits=logits)
            action = action_dist.sample()
            log_p = action_dist.log_prob(action).item()

            ac = action.cpu().item()
            next_state, reward, done, trunc, _ = self.env.step(ac)

            entropy = action_dist.entropy().item()

            states.append(state)
            actions.append(ac)
            logps.append(log_p)
            entropies.append(entropy)

            # Group by episode
            ep_rewards.append(reward)
            ep_values.append(critic_value.item())
            ep_entropies.append(entropy)

            state = next_state
            episode_step += 1

            iter_end = step >= (self.steps_per_iteration - 1)
            terminate = (
                episode_step >= self.steps_per_episode
                or iter_end
                or done
                or trunc
            )

            if terminate:
                # If truncated, guess the value of the "final" state
                if not done:
                    final_val = self._agent(s)[1].item()
                else:
                    final_val = 0

                # log episode rewards and length
                ep_rewards = ep_rewards + [final_val]
                ep_values = ep_values + [final_val]
                rets = self.compute_return(ep_rewards)[:-1]
                target_values.extend(rets)
                # compute returns, remember last one is dummy
                avg_rewards.append(sum(ep_rewards[:-1]))
                # compute advantages
                advs = self.compute_advantage(ep_rewards, ep_values)
                advantages.extend(advs)
                avg_adv = sum(advs) / len(advs)
                avg_advantages.append(avg_adv)
                # log entropies for batch
                avg_entropy = sum(ep_entropies) / len(ep_entropies)
                avg_entropies.append(avg_entropy)
                # reset
                state, _ = self.env.reset()
                episode_step = 0
                ep_rewards.clear()
                ep_values.clear()
                ep_entropies.clear()

            if iter_end:
                iter_len = self.steps_per_iteration
                num_episodes = len(avg_rewards)
                avg_ep_len = iter_len / num_episodes

        return {
            'data': {
                'states': states,
                'actions': actions,
                'logps': logps,
                'target_values': target_values,
                'advantages': advantages,
                'entropies': entropies,
            },
            'stats': {
                'avg_entropy': avg_entropies,
                'avg_ep_reward': avg_rewards,
                'avg_qvalue': avg_advantages,
                'avg_ep_len': avg_ep_len,
            }
        }