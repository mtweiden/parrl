"""Implementation of ReplayBuffers for DVNs."""
from __future__ import annotations

from collections import deque

from numpy import ndarray
from numpy import stack
from numpy.random import choice
from numpy.random import uniform

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

    def store(self, s: ndarray) -> None:
        """
        Store a single state.

        Args:
            s (ndarray): Current state.
        """
        self.buffer.append(s)

    def sample_batch(self, batch_size: int) -> ndarray :
        indices = choice(len(self), size=batch_size, replace=False)
        batch = []
        for i in indices:
            batch.append(self.buffer[i])
        states = stack(batch)
        return states

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A ReplayBuffer that does Prioritized Experience Replay.
    """
    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        max_priority: float = 1.0,
    ) -> None:
        super(PrioritizedReplayBuffer, self).__init__(max_size)

        self.max_priority = max_priority
        self.tree_ptr = 0
        self.alpha = alpha

        tree_size = 1
        while tree_size < self.max_size:
            tree_size *= 2

        self.min_tree = MinSegmentTree(tree_size)
        self.sum_tree = SumSegmentTree(tree_size)

    def store(self, s: ndarray) -> None:
        super().store(s)
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample(self, beta: float = 0.4) -> tuple[ndarray, float, int]:
        assert beta > 0
        index = self._sample_proportional(num_samples=1)[0]
        s, weight = self.sample_from_index(index, beta)
        return s, weight, index

    def sample_from_index(
        self,
        index: int,
        beta: float = 0.4,
    ) -> tuple[ndarray, float]:
        assert beta > 0
        s = self.buffer[index]
        weight = self._calculate_weight(index, beta)
        samp = s, weight
        return samp  # type: ignore

    def sample_batch(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> tuple[ndarray, ...]:
        assert beta > 0
        indices = self._sample_proportional(num_samples=batch_size)
        states, weights = [], []
        for i in indices:
            s, w = self.sample_from_index(i, beta)
            states.append(s)
            weights.append(w)
        states = stack(states)
        weights = stack(weights)
        indices = stack(indices)
        batch = states, weights, indices
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

    def clear(self) -> None:
        self.buffer.clear()
        self.min_tree.clear()
        self.sum_tree.clear()
