from typing import Any
from typing import Sequence

from collections import deque

from numpy import copy
from numpy import stack
from numpy import ndarray

from random import sample as random_sample
from random import seed as random_seed

from torch import tensor
from torch import Tensor
from torch.utils.data import Dataset


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        *,
        seed: int | None = None
    ) -> None:
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        if seed is not None:
            random_seed(seed)
    
    def clear(self) -> None:
        self.buffer.clear()

    def sample(self) -> Any:
        return random_sample(self.buffer, 1)

    def sample_batch(self, num_samples: int) -> list[Any]:
        _batch = random_sample(self.buffer, num_samples)
        separate_lists = [[] for _ in range(len(_batch[0]))]
        for items in _batch:
            for x, _list in zip(items, separate_lists):
                _list.append(x)
        batch = [tensor(stack(_list)) for _list in separate_lists]
        return batch

    def store(self, experience: Sequence[Any]) -> None:
        for transition in experience:
            modified_transition = []
            # PyTorch wants writable arrays
            for x in transition:
                if isinstance(x, ndarray):
                    x = copy(x)
                modified_transition.append(x)
            self.add(*tuple(modified_transition))

    def add(self, *args) -> None:
        self.buffer.append(args)

    def __len__(self) -> int:
        return len(self.buffer)
    
    @staticmethod
    def group_data(data_dict: dict[str, Sequence]) -> list[tuple]:
        """Turns a dict of results into a list of experiences."""
        data_seqs = [data_dict[key] for key in data_dict]
        data_list = [data for data in zip(*data_seqs)]
        return data_list


class ReplayBufferDataset(Dataset):
    """
    A Dataset that draws data by sampling from a ReplayBuffer.
    """
    def __init__(
        self,
        buffer: ReplayBuffer,
    ) -> None:
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer)
    
    def __getitem__(self, idx: int) -> tuple[Tensor]:
        sample = self.buffer.buffer[idx]
        return sample
