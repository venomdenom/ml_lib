from typing import Optional, Dict

import numpy as np

from gradient_descent.batch import DataBatcher
from gradient_descent.loss_functions import LogisticLoss, LossFunction
from gradient_descent.optimizers import GradientDescentOptimizer, SGD, GradientInfo
from gradient_descent.trainer import GradientDescentTrainer
from regulizers.regulizers import Regularizer


class LogisticRegression(GradientDescentTrainer):
    def __init__(
            self,
            optimizer: Optional[GradientDescentOptimizer] = None,
            regularizer: Optional[Regularizer] = None,
            loss_function: Optional[LossFunction] = LogisticLoss,
            max_epochs: int = 1000,
            tolerance: float = 1e-6,
            verbose: bool = True
    ):
        if optimizer is None:
            optimizer = SGD(learning_rate=0.01)

        super().__init__(
            loss_function=loss_function(),
            optimizer=optimizer,
            max_epochs=max_epochs,
            tolerance=tolerance,
            verbose=verbose
        )

        self.regularizer = regularizer


    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))


    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        linear_output = X @ self.weights
        return self._sigmoid(linear_output)


    def fit(
            self, X: np.ndarray, y: np.ndarray,
            batch_size: Optional[int] = None,
            initial_weights: Optional[np.ndarray] = None
    ) -> 'LogisticRegression':

        X_with_bias = np.column_stack([np.ones(len(X)), X])

        if initial_weights is None:
            self.weights = np.random.randn(X_with_bias.shape[1]) * 0.01
        else:
            self.weights = initial_weights.copy()

        batcher = DataBatcher(X_with_bias, y, batch_size)

        self.optimizer.reset()

        prev_loss = float('inf')

        for epoch in range(self.max_epochs):
            epoch_losses = []

            for batch in batcher:
                y_pred = self._predict_batch(batch.X)

                base_loss = self.loss_function.loss(batch.y, y_pred)
                base_gradient = self.loss_function.gradient(batch.y, y_pred, batch.X)

                total_loss = base_loss
                total_gradient = base_gradient

                if self.regularizer is not None:
                    reg_penalty = self.regularizer.penalty(self.weights)
                    reg_gradient = self.regularizer.gradient(self.weights)

                    total_loss += reg_penalty
                    total_gradient += reg_gradient

                grad_info = GradientInfo(
                    gradient=total_gradient,
                    loss=total_loss,
                    batch_size=batch.size,
                    iteration=epoch
                )

                self.weights = self.optimizer.update(self.weights, grad_info)
                epoch_losses.append(total_loss)

            avg_loss = np.mean(epoch_losses)
            self.training_history.append(avg_loss)

            if abs(prev_loss - avg_loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at epoch {epoch}, loss: {avg_loss:.6f}")
                break

            prev_loss = avg_loss

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return self._predict_batch(X_with_bias)


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


    def get_training_metadata(self) -> Dict[str, np.ndarray]:
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        return {
            'intercept': self.weights[0],
            'coefficients': self.weights[1:],
            'all_weights': self.weights
        }
