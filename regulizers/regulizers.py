from abc import ABC, abstractmethod

import numpy as np


class Regularizer(ABC):

    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        pass


    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        pass


class L2Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha


    def penalty(self, weights: np.ndarray) -> float:
        return self.alpha * np.sum(weights[1:] ** 2)


    def gradient(self, weights: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(weights)
        grad[1:] = 2 * self.alpha * weights[1:]
        return grad


class L1Regularizer(Regularizer):

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha


    def penalty(self, weights: np.ndarray) -> float:
        return self.alpha * np.sum(np.abs(weights[1:]))


    def gradient(self, weights: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(weights)
        grad[1:] = self.alpha * np.sign(weights[1:])
        return grad
