"""
Implementation of pytorch Datasets for ReplayBuffers for DVNs.
"""
from __future__ import annotations

from numpy import ndarray

from torch.utils.data import Dataset

from parrl.buffers.multistep_dvn_buffer import ReplayBuffer
from parrl.buffers.multistep_dvn_buffer import PrioritizedReplayBuffer


class ReplayDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[ndarray, bool, float]:
        # Ignore index and sample from the buffer
        s, d, r = self.buffer.buffer[index]
        return s, d, r


class PrioritizedReplayDataset(Dataset):
    def __init__(self, buffer: PrioritizedReplayBuffer) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[ndarray, bool, float, float, int]: 
        # Ignore index and sample from the buffer
        s, d, r, w, i = self.buffer.sample(beta=0.4)
        return s, d, r, w, i
