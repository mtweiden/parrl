"""
Segement tree implementation for Prioritized Experience Replay.

Based on: https://github.com/openai/baselines/blob/master/baselines \
    /common/segment_tree.py
"""
from typing import Callable

from operator import add


class SegmentTree:
    def __init__(
        self,
        capacity: int,
        operation: Callable,
        neutral_value: float,
    ) -> None:
        m = "capacity must be positive and a power of 2."
        assert capacity > 0 and capacity & (capacity - 1) == 0, m
        self._capacity = capacity
        self._value = [neutral_value for _ in range(2 * capacity)]
        self._operation = operation

    def clear(self) -> None:
        self._value.clear()

    def _reduce_helper(
        self,
        start: int,
        end: int,
        node: int,
        node_start: int,
        node_end: int,
    ) -> float:
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start: int=0, end:int | None=None) -> float:
        """
        Returns result of applying `self.operation` to a contiguous subsequence 
        of the array.

        Args:
            start (int): The beginning of the subsequence.
            end (int): The end of the subsequences.

        Returns:
            reduced (float): The result of reducing self.operation over the 
                specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx) -> float:
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super(SumSegmentTree, self).__init__(capacity, add, 0.0)

    def sum(self, start: int=0, end: int | None=None) -> float:
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function allows to sample 
        indexes according to the discrete probability efficiently.

        Args:
            perfixsum (float): Upperbound on the sum of array prefix.

        Returns:
            idx (int): Highest index satisfying the prefixsum constraint.
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super(MinSegmentTree, self).__init__(capacity, min, float('inf'))

    def min(self, start: int=0, end: int | None=None) -> float:
        """Returns min(arr[start], ...,  arr[end])"""
        return super(MinSegmentTree, self).reduce(start, end)
