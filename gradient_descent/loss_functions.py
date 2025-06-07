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


class MeanAbsoluteError(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        signs = np.sign(y_pred - y_true)
        return 1 / n * X.T @ signs


class HuberLoss(LossFunction):

    def __init__(self, delta: float = 1.0):
        self.delta = delta


    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residual = np.abs(y_pred - y_true)
        condition = residual <= self.delta
        squared_loss = 0.5 * (y_pred - y_true) ** 2
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        return float(np.mean(np.where(condition, squared_loss, linear_loss)))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        residual = y_pred - y_true
        condition = np.abs(residual) <= self.delta
        grad_residual = np.where(condition, residual, self.delta * np.sign(residual))
        return 1 / n * X.T @ grad_residual


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


class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        return 1 / n * X.T @ (y_pred - y_true)


class SparseCategoricalCrossEntropy(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        correct_class_probs = y_pred[np.arange(len(y_true)), y_true.astype(int)]
        return float(-np.mean(np.log(correct_class_probs)))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        # Создаем one-hot encoding
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
        return 1 / n * X.T @ (y_pred - y_one_hot)


class FocalLoss(LossFunction):

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma


    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        ce_loss = -np.log(p_t)
        return float(np.mean(focal_weight * ce_loss))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
        one_minus_p_t = 1 - p_t

        focal_weight = self.alpha * (one_minus_p_t ** self.gamma)

        grad_focal = focal_weight * (self.gamma * np.log(p_t) * one_minus_p_t + 1)

        grad_sign = np.where(y_true == 1, -1, 1)
        grad_residual = grad_sign * grad_focal

        return 1 / n * X.T @ grad_residual


class HingeLoss(LossFunction):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        margin = 1 - y_true * y_pred
        return float(np.mean(np.maximum(0, margin)))


    def gradient(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            X: np.ndarray
    ) -> np.ndarray:
        n = len(y_true)
        margin = 1 - y_true * y_pred
        grad_mask = (margin > 0).astype(float)
        grad_residual = -y_true * grad_mask
        return 1 / n * X.T @ grad_residual
