"""
Implementation of ReplayBuffers for DQNs.
Those that do multi-step returns are based off:
    https://github.com/Curt-Park/rainbow-is-all-you-need.
"""
from __future__ import annotations

from collections import deque

from numpy import array
from numpy import float32
from numpy import ndarray
from numpy import stack
from numpy import zeros
from numpy.random import choice
from numpy.random import uniform

from torch.utils.data import Dataset

from typing import Sequence

from parrl.core.utils.segment_tree import MinSegmentTree
from parrl.core.utils.segment_tree import SumSegmentTree


class ReplayBuffer:
    """
    A simple ReplayBuffer implemented with a deque.
    """
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def store(
        self,
        s: ndarray,
        a: ndarray,
        r: float,
        ns: ndarray,
        done: bool,
    ) -> tuple[ndarray, ndarray, float, ndarray, bool]:
        """
        Store a single transition.

        Args:
            s (ndarray): Current state.

            a (ndarray): Action taken.

            ns (ndarray): Next state.

            r (float): Reward.

            done (bool): Done flag.
        """
        transition = (s, a, r, ns, done)
        self.buffer.append(transition)
        return transition

    def sample_batch(self, batch_size: int) -> tuple[ndarray, ...]:
        indices = choice(len(self), size=batch_size, replace=False)
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        states = stack([b[0] for b in batch])
        actions = stack([b[1] for b in batch])
        rewards = stack([b[2] for b in batch])
        next_states = stack([b[3] for b in batch])
        dones = stack([b[4] for b in batch])
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A ReplayBuffer that does Prioritized Experience Replay.
    """
    def __init__(self, max_size: int, alpha: float = 0.6) -> None:
        super(PrioritizedReplayBuffer, self).__init__(max_size)

        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        tree_size = 1
        while tree_size < self.max_size:
            tree_size *= 2

        self.min_tree = MinSegmentTree(tree_size)
        self.sum_tree = SumSegmentTree(tree_size)

    def store(
        self,
        s: ndarray,
        a: ndarray,
        r: float,
        ns: ndarray,
        done: bool,
    ) -> tuple[ndarray, ndarray, float, ndarray, bool] | tuple:
        transition = super().store(s, a, r, ns, done)
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

    def sample(
        self,
        beta: float = 0.4,
    ) -> tuple[ndarray, int, float, ndarray, bool, float, int]: 
        assert beta > 0
        index = self._sample_proportional(num_samples=1)[0]
        samp = self.sample_from_index(index, beta)
        return *samp, index  # type: ignore

    def sample_from_index(
        self,
        index: int,
        beta: float = 0.4,
    ) -> tuple[ndarray, int, float, ndarray, bool, float]: 
        assert beta > 0
        s, a, r, ns, d = self.buffer[index]
        weight = self._calculate_weight(index, beta)
        samp = s, a, r, ns, d, weight
        return samp  # type: ignore

    def sample_batch(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> tuple[ndarray, ...]:
        assert beta > 0
        indices = self._sample_proportional(num_samples=batch_size)
        states, acs, rews, nstates, dones, weights = [], [], [], [], [], []
        for i in indices:
            s, a, r, ns, d, w = self.sample_from_index(i, beta)
            states.append(s)
            acs.append(a)
            rews.append(r)
            nstates.append(ns)
            dones.append(d)
            weights.append(w)
        states = stack(states)
        acs = stack(acs)
        rews = stack(rews)
        nstates = stack(nstates)
        dones = stack(dones)
        weights = stack(weights)
        indices = stack(indices)
        batch = states, acs, rews, nstates, dones, weights, indices
        return batch  # type: ignore

    def update_priorities(
        self,
        indices: Sequence[int],
        priorities: ndarray,
    ) -> None:
        """Update sample priority values."""
        assert len(indices) == len(priorities)
        for i, p in zip(indices, priorities):
            assert p > 0
            assert 0 <= i < len(self)
            self.min_tree[i] = p ** self.alpha
            self.sum_tree[i] = p ** self.alpha
            self.max_priority = max(self.max_priority, p)

    def _sample_proportional(self, num_samples: int) -> list[int]:
        """Sample buffer indices based on priorities."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / num_samples
        for i in range(num_samples):
            a = segment * i
            b = segment * (i + 1)
            upperbound = uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, index: int, beta: float) -> float:
        # Get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        # Compute weights
        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight


class MultiStepReplayBuffer:
    """
    A ReplayBuffer that uses multi-step returns.
    """
    def __init__(
        self,
        state_dim: int,
        size: int,
        batch_size: int,
        num_steps: int = 1,
        gamma: float = 0.998,
    ) -> None:
        self.s_buf = zeros([size, state_dim], dtype=float32)
        self.ns_buf = zeros([size, state_dim], dtype=float32)
        self.a_buf = zeros([size], dtype=float32)
        self.r_buf = zeros([size], dtype=float32)
        self.d_buf = zeros([size], dtype=float32)

        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        # For multi-step learning
        self.n_step_buf = deque(maxlen=num_steps)
        self.num_steps = num_steps
        self.gamma = gamma

    def store(
        self,
        s: ndarray,
        a: ndarray,
        r: float,
        ns: ndarray,
        done: bool,
        trunc: bool,
    ) -> tuple[ndarray, ndarray, float, ndarray, bool] | tuple:
        """
        Store a single transition.

        Args:
            s (ndarray): Current state.

            a (ndarray): Action taken.

            ns (ndarray): Next state.

            r (float): Reward.

            done (bool): Done flag.

            trunc (bool): Truncation flag. Means that the trajectory was
                ended but a terminal state was not reached.
        """
        t = (s, a, r, ns, done, trunc)
        self.n_step_buf.append(t)

        # Still waiting for more transitions to do multi-step learning
        if len(self.n_step_buf) < self.num_steps:
            return ()

        # Do multi-step learning
        r, ns, d = self._get_multi_step(self.n_step_buf, self.gamma)
        s, a = self.n_step_buf[0][:2]
        
        self.s_buf[self.ptr] = s
        self.ns_buf[self.ptr] = ns
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.d_buf[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buf[0]

    def sample_batch(self) -> tuple[ndarray, ...]:
        indices = choice(self.size, size=self.batch_size, replace=False)
        return self.sample_batch_from_indices(indices)

    def sample_batch_from_indices(
        self,
        indices: ndarray,
    ) -> tuple[ndarray, ...]:
        s = self.s_buf[indices],
        a = self.a_buf[indices],
        ns = self.ns_buf[indices],
        r = self.r_buf[indices],
        d = self.d_buf[indices],
        batch = s, a, r, ns, d, indices
        return batch

    def _get_multi_step(
        self,
        buf: deque,
        gamma: float
    ) -> tuple[ndarray, ndarray, bool]:
        """Compute the multi-step return."""
        reward, next_state, done, truncated = buf[-1][-4:]

        if done or truncated:
            return reward, next_state, done

        for transition in reversed(list(buf)[:-1]):
            r, ns, d, trunc = transition[-4:]
            reward = r + gamma * reward * (1 - d)
            next_state, done = (ns, d) if d or trunc else (next_state, done)

        return reward, next_state, done

    def __len__(self) -> int:
        return self.size


class PrioritizedMultiStepReplayBuffer(MultiStepReplayBuffer):
    """
    A ReplayBuffer that does Prioritized Experience Replay and computes
    multi-step returns.
    """

    def __init__(
        self,
        state_dim: int,
        size: int,
        batch_size: int,
        num_steps: int = 1,
        gamma: float = 0.998,
        alpha: float = 0.6,
    ) -> None:
        args = (state_dim, size, batch_size, num_steps, gamma)
        super(PrioritizedMultiStepReplayBuffer, self).__init__(*args)

        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        tree_size = 1
        while tree_size < self.max_size:
            tree_size *= 2

        self.min_tree = MinSegmentTree(tree_size)
        self.sum_tree = SumSegmentTree(tree_size)

    def store(
        self,
        s: ndarray,
        a: ndarray,
        r: float,
        ns: ndarray,
        done: bool,
        trunc: bool,
    ) -> tuple[ndarray, ndarray, float, ndarray, bool] | tuple:
        # Do regular store
        transition = super().store(s, a, r, ns, done, trunc)
        # If n-step return was computed, update trees
        if transition:
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

    def sample_batch(self, beta: float = 0.4) -> tuple[ndarray, ...]:
        assert len(self) >= self.batch_size
        assert beta > 0
        indices = self._sample_proportional()
        s = self.s_buf[indices],
        a = self.a_buf[indices],
        ns = self.ns_buf[indices],
        r = self.r_buf[indices],
        d = self.d_buf[indices],
        weights = array(
            [self._calculate_weight(i, beta) for i in indices],
            dtype=float32,
        )
        batch = s, a, r, ns, d, weights, indices
        return batch

    def update_priorities(
        self,
        indices: Sequence[int],
        priorities: ndarray,
    ) -> None:
        """Update sample priority values."""
        assert len(indices) == len(priorities)
        for i, p in zip(indices, priorities):
            assert p > 0
            assert 0 <= i < len(self)
            self.min_tree[i] = p ** self.alpha
            self.sum_tree[i] = p ** self.alpha
            self.max_priority = max(self.max_priority, p)

    def _sample_proportional(self) -> list[int]:
        """Sample buffer indices based on priorities."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, index: int, beta: float) -> float:
        # Get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        # Compute weights
        p_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        return weight
