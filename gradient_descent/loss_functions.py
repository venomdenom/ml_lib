from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


    @abstractmethod
    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        pass


class MeanSquaredError(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        return -2 / n * X.T @ (y_true - y_pred)


class LogisticLoss(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(
            -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        return 1 / n * X.T @ (y_pred - y_true)
