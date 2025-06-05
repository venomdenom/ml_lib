from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GradientInfo:
    gradient: np.ndarray
    loss: float
    batch_size: int
    iteration: int


class GradientDescentOptimizer(ABC):

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.history: List[float] = []


    @abstractmethod
    def update(self, weights: np.ndarray, grad_info: GradientInfo) -> np.ndarray:
        pass


    def reset(self) -> None:
        self.history = []


class SGD(GradientDescentOptimizer):

    def update(self, weights: np.ndarray, grad_info: GradientInfo) -> np.ndarray:
        self.history.append(grad_info.loss)
        return weights - self.learning_rate * grad_info.gradient


class MomentumSGD(GradientDescentOptimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None


    def update(self, weights: np.ndarray, grad_info: GradientInfo) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        self.velocity = (self.momentum * self.velocity + self.learning_rate *
                         grad_info.gradient)
        self.history.append(grad_info.loss)
        return weights - self.velocity


    def reset(self) -> None:
        self.velocity = None
        super().reset()


class AdaGrad(GradientDescentOptimizer):

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.sum_squared_gradients = None


    def update(self, weights: np.ndarray, grad_info: GradientInfo) -> np.ndarray:
        if self.sum_squared_gradients is None:
            self.sum_squared_gradients = np.zeros_like(weights)

        self.sum_squared_gradients += grad_info.gradient ** 2
        adapted_lr = self.learning_rate / (
                np.sqrt(self.sum_squared_gradients) + self.epsilon)

        self.history.append(grad_info.loss)
        return weights - adapted_lr * grad_info.gradient


    def reset(self) -> None:
        self.sum_squared_gradients = None
        super().reset()


class Adam(GradientDescentOptimizer):

    def __init__(
            self, learning_rate: float = 0.001, beta1: float = 0.9,
            beta2: float = 0.999, epsilon: float = 1e-8
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0  # Time step


    def update(self, weights: np.ndarray, grad_info: GradientInfo) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_info.gradient

        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_info.gradient ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update weights
        self.history.append(grad_info.loss)
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0
        super().reset()
