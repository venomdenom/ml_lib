from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from gradient_descent.batch import DataBatcher
from gradient_descent.loss_functions import LossFunction
from gradient_descent.optimizers import GradientDescentOptimizer, GradientInfo


class GradientDescentTrainer(ABC):

    def __init__(
            self,
            loss_function: LossFunction,
            optimizer: GradientDescentOptimizer,
            max_epochs: int = 1000,
            tolerance: float = 1e-6,
            verbose: bool = True
    ) -> None:
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.verbose = verbose

        self.weights = None
        self.training_history = []


    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: Optional[int] = None,
            initial_weights: Optional[np.ndarray] = None,
    ) -> 'GradientDescentTrainer':
        X_with_bias = np.column_stack([np.ones(len(X)), X])

        if initial_weights is None:
            self.weights = np.random.randn(X_with_bias.shape[1]) * 0.01
        else:
            self.weights = initial_weights

        batcher = DataBatcher(X_with_bias, y, batch_size)

        self.optimizer.reset()

        previous_loss = np.inf

        for epoch in range(self.max_epochs):
            epoch_losses = []
            for batch in batcher:
                y_pred = self._predict_batch(batch.X)

                loss = self.loss_function.loss(batch.y, y_pred)
                gradient = self.loss_function.gradient(batch.y, y_pred, batch.X)

                gradient_info = GradientInfo(
                    gradient=gradient,
                    loss=loss,
                    batch_size=batch.size,
                    iteration=epoch
                )

                self.weights = self.optimizer.update(self.weights, gradient_info)
                epoch_losses.append(loss)

            average_loss = np.mean(epoch_losses)
            self.training_history.append(average_loss)

            if abs(previous_loss - average_loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at epoch {epoch}, loss: {average_loss:.6f}")
                break

            previous_loss = average_loss
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, loss: {average_loss:.6f}")

        return self

    @abstractmethod
    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        pass


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been trained yet")

        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return self._predict_batch(X_with_bias)
