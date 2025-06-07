from typing import Optional, Dict, Any

import numpy as np

from gradient_descent.batch import DataBatcher
from gradient_descent.loss_functions import LossFunction, HingeLoss
from gradient_descent.optimizers import GradientDescentOptimizer, SGD, GradientInfo
from gradient_descent.trainer import GradientDescentTrainer
from regulizers.regulizers import Regularizer, L2Regularizer


class MulticlassSVM:
    """One vs all SVM"""


    def __init__(
            self,
            optimizer: Optional[GradientDescentOptimizer] = None,
            regularizer: Optional[Regularizer] = None,
            loss_function: Optional[LossFunction] = HingeLoss,
            max_epochs: int = 1000,
            tolerance: float = 1e-6,
            verbose: bool = True,
            C: float = 1.0
    ) -> None:
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.loss_function = loss_function
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.C = C
        self.classes_ = None
        self.classifiers_ = {}


    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MulticlassSVM':
        self.classes_ = np.unique(y)
        assert len(self.classes_) >= 1, "I really consider you a very bad person"

        if len(self.classes_) == 2:
            svm = SVM(
                optimizer=self.optimizer,
                regularizer=self.regularizer,
                loss_function=self.loss_function,
                max_epochs=self.max_epochs,
                tolerance=self.tolerance,
                verbose=self.verbose,
                C=self.C
            )
            svm.fit(X, y, **kwargs)
            self.classifiers_[f"{self.classes_[0]}_vs_rest"] = svm
        else:
            for class_label in self.classes_:
                if self.verbose:
                    print(f"Training classifier for class {class_label}")

                y_binary = np.where(y == class_label, 1, 0)
                svm = SVM(
                    optimizer=self.optimizer,
                    regularizer=self.regularizer,
                    loss_function=self.loss_function,
                    max_epochs=self.max_epochs,
                    tolerance=self.tolerance,
                    verbose=False,
                    C=self.C
                )
                svm.fit(X, y_binary, **kwargs)
                self.classifiers_[f"{class_label}_vs_rest"] = svm

        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self.classes_) == 2:
            # Бинарная классификация
            classifier = list(self.classifiers_.values())[0]
            predictions = classifier.predict(X)
            return np.where(predictions == 1, self.classes_[1], self.classes_[0])
        else:
            decision_scores = np.zeros((len(X), len(self.classes_)))

            for i, class_label in enumerate(self.classes_):
                classifier = self.classifiers_[f"{class_label}_vs_rest"]
                scores = classifier.decision_function(X)
                decision_scores[:, i] = scores

            predicted_indices = np.argmax(decision_scores, axis=1)
            return self.classes_[predicted_indices]


    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if len(self.classes_) == 2:
            classifier = list(self.classifiers_.values())[0]
            return classifier.decision_function(X)
        else:
            decision_scores = np.zeros((len(X), len(self.classes_)))
            for i, class_label in enumerate(self.classes_):
                classifier = self.classifiers_[f"{class_label}_vs_rest"]
                decision_scores[:, i] = classifier.decision_function(X)
            return decision_scores


class SVM(GradientDescentTrainer):
    def __init__(
            self,
            optimizer: Optional[GradientDescentOptimizer] = None,
            regularizer: Optional[Regularizer] = None,
            loss_function: Optional[LossFunction] = HingeLoss,
            max_epochs: int = 1000,
            tolerance: float = 1e-6,
            verbose: bool = True,
            C: float = 1.0  # Regularization parameter
    ):
        if optimizer is None:
            optimizer = SGD(learning_rate=0.01)

        if regularizer is None:
            regularizer = L2Regularizer(alpha=1.0 / C)

        super().__init__(
            loss_function=loss_function(),
            optimizer=optimizer,
            max_epochs=max_epochs,
            tolerance=tolerance,
            verbose=verbose
        )

        self.regularizer = regularizer
        self.C = C


    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


    def _prepare_labels(self, y: np.ndarray) -> np.ndarray:
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(
                f"SVM supports only binary classification. Found "
                f"{len(unique_labels)}. For multi-class classification, "
                f"use MulticlassSVM."
            )

        y_transformed = np.where(y == unique_labels[0], -1, 1)
        return y_transformed


    def fit(
            self, X: np.ndarray, y: np.ndarray,
            batch_size: Optional[int] = None,
            initial_weights: Optional[np.ndarray] = None
    ) -> 'SVM':
        y_transformed = self._prepare_labels(y)
        X_with_bias = np.column_stack([np.ones(len(X)), X])

        if initial_weights is None:
            self.weights = np.random.randn(X_with_bias.shape[1]) * 0.01
        else:
            self.weights = initial_weights.copy()

        batcher = DataBatcher(X_with_bias, y_transformed, batch_size)

        self.optimizer.reset()

        previous_loss = float('inf')

        for epoch in range(self.max_epochs):
            epoch_losses = []
            for batch in batcher:
                y_pred = self._predict_batch(batch)

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

            average_loss = np.mean(epoch_losses)
            self.training_history.append(average_loss)

            if abs(average_loss - previous_loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at epoch {epoch}, loss: {average_loss:.2f}")
                break

            previous_loss = average_loss
            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {average_loss:.4f}")

        return self


    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        X_with_bias = np.column_stack([np.ones(len(X)), X])
        return self._predict_batch(X_with_bias)


    def predict(self, X: np.ndarray) -> np.ndarray:
        decision_scores = self.decision_function(X)
        return np.where(decision_scores >= 0, 1, 0)


    def get_support_vectors_info(
            self,
            X: np.ndarray,
            y: np.ndarray,
            margin_tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        y_transformed = self._prepare_labels(y)
        decision_scores = self.decision_function(X)

        margins = y_transformed * decision_scores

        support_mask = margins <= (1 + margin_tolerance)

        on_margin_mask = np.abs(margins - 1) <= margin_tolerance

        inside_margin_mask = margins < (1 - margin_tolerance)

        return {
            'support_vectors': X[support_mask],
            'support_vector_labels': y[support_mask],
            'support_vector_indices': np.where(support_mask)[0],
            'n_support_vectors': np.sum(support_mask),
            'on_margin_vectors': X[on_margin_mask],
            'inside_margin_vectors': X[inside_margin_mask],
            'margins': margins
        }


    def get_training_metadata(self) -> Dict[str, np.ndarray]:
        if self.weights is None:
            raise ValueError("Model not trained yet!")

        return {
            'intercept': self.weights[0],
            'coefficients': self.weights[1:],
            'all_weights': self.weights,
            'C': self.C
        }
