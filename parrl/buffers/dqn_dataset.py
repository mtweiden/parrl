"""
Implementation of pytorch Datasets for ReplayBuffers for DQNs.
"""
from __future__ import annotations

from numpy import ndarray

from torch.utils.data import Dataset

from parrl.buffers.dqn_buffer import ReplayBuffer
from parrl.buffers.dqn_buffer import PrioritizedReplayBuffer


class ReplayDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[ndarray, int, float, ndarray, bool]:
        # Ignore index and sample from the buffer
        s, a, r, ns, d = self.buffer.buffer[index]
        return s, a, r, ns, d


class PrioritizedReplayDataset(Dataset):
    def __init__(self, buffer: PrioritizedReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[ndarray, int, float, ndarray, bool, float, int]: 
        # Ignore index and sample from the buffer
        s, a, r, ns, d, w, i = self.buffer.sample(beta=0.4)
        return s, a, r, ns, d, w, i
