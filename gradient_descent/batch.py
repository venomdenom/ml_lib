from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

import numpy as np


class BatchType(StrEnum):
    FULL = 'full'
    MINI = 'mini'
    STOCHASTIC = 'stochastic'


@dataclass
class Batch:
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    batch_type: BatchType


    @property
    def size(self) -> int:
        return len(self.X)


    def __iter__(self):
        return iter((self.X, self.y))


class DataBatcher:

    def __init__(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            batch_size: Optional[int] = None,
            shuffle: bool = True
    ) -> None:
        self.X = X
        self.Y = Y
        self.n_samples = len(X)
        self.shuffle = shuffle

        if batch_size is None:
            self.batch_size = self.n_samples
            self.batch_type = BatchType.FULL
        elif batch_size == 1:
            self.batch_size = 1
            self.batch_type = BatchType.STOCHASTIC
        else:
            self.batch_size = batch_size
            self.batch_type = BatchType.MINI


    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_indices = indices[start:end]

            yield Batch(
                self.X[batch_indices],
                self.Y[batch_indices],
                indices=batch_indices,
                batch_type=self.batch_type
            )
