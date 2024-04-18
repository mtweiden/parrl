"""
Implementation of pytorch Datasets for ReplayBuffers for DVNs.
"""
from __future__ import annotations

from numpy import ndarray

from torch.utils.data import Dataset

from parrl.buffers.dvn_buffer import ReplayBuffer
from parrl.buffers.dvn_buffer import PrioritizedReplayBuffer


class ReplayDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> ndarray:
        # Ignore index and sample from the buffer
        s = self.buffer.buffer[index]
        return s


class PrioritizedReplayDataset(Dataset):
    def __init__(self, buffer: PrioritizedReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[ndarray, float, int]: 
        # Ignore index and sample from the buffer
        s, w, i = self.buffer.sample(beta=0.4)
        return s, w, i
