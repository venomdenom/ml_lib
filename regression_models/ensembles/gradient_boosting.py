from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Any, Generator, Dict

import numpy as np
from numpy import ndarray, dtype, generic

from regression_models.tree.criterions import ImpurityCriterion, MSECriterion
from regression_models.tree.decision_tree import DecisionTreeRegressor


class GBDTLoss(ABC):

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


    @abstractmethod
    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


    @abstractmethod
    def init_estimate(self, y: np.ndarray) -> float:
        pass


class MSELossGBDT(GBDTLoss):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))


    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # For MSE, negative gradient is simply the residuals
        return y_true - y_pred


    def init_estimate(self, y: np.ndarray) -> float:
        return np.mean(y)


class MAELossGBDT(GBDTLoss):

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_pred - y_true)))


    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # For MAE, negative gradient is the sign of residuals
        return np.sign(y_true - y_pred)


    def init_estimate(self, y: np.ndarray) -> float:
        return np.median(y)


class HuberLossGBDT(GBDTLoss):

    def __init__(self, delta: float = 1.0) -> None:
        self.delta = delta


    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residual = np.abs(y_true - y_pred)
        return float(
            np.mean(
                np.where(
                    residual <= self.delta,
                    0.5 * residual ** 2,
                    self.delta * residual - 0.5 * self.delta ** 2
                )
            )
        )


    def negative_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        residual = y_true - y_pred
        return np.where(
            np.abs(residual) <= self.delta,
            residual,
            self.delta * np.sign(residual)
        )


    def init_estimate(self, y: np.ndarray) -> float:
        return np.mean(y)


class GradientBoostingRegressor:

    def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: int = 3,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_impurity_decrease: float = 0.0,
            subsample: float = 1.0,
            max_features: Union[str, int, float, None] = 'sqrt',
            loss: GBDTLoss = MSELossGBDT,
            criterion: ImpurityCriterion = MSECriterion,
            random_state: Optional[int] = None,
            verbose: bool = False,
            validation_fraction: float = 0.1,
            n_iter_no_change: Optional[int] = None,
            tolerance: float = 1e-4
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample = subsample
        self.max_features = max_features
        self.loss_function = loss()
        self.criterion = criterion
        self.random_state = random_state
        self.verbose = verbose
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tolerance = tolerance

        # Model state
        self.estimators_ = []
        self.init_ = None
        self.feature_importances_ = None
        self.train_score_ = []
        self.validation_score_ = []
        self.n_features_ = None
        self._n_iter = 0

        if random_state is not None:
            np.random.seed(random_state)


    def _get_max_features(self, n_features: int) -> Optional[int]:
        if self.max_features is None:
            return None
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                return int(np.log2(n_features))
            elif self.max_features == 'auto':
                return int(np.sqrt(n_features))
            else:
                raise ValueError(f"Unknown max_features: {self.max_features}")
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        else:
            return n_features


    def _create_estimator(self) -> DecisionTreeRegressor:
        max_features = self._get_max_features(self.n_estimators)

        return DecisionTreeRegressor(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=max_features,
            random_state=None
        )


    def _subsample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.subsample >= 1:
            return X, y

        n_samples = X.shape[0]
        n_subsample = max(1, int(self.subsample * n_samples))

        indices = np.random.choice(n_samples, size=n_subsample, replace=False)
        return X[indices], y[indices]


    def _train_validation_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if self.validation_fraction <= 0 or self.n_iter_no_change is None:
            return X, y, None, None

        n_samples = X.shape[0]
        n_validation = max(1, int(self.validation_fraction * n_samples))

        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_validation]
        train_indices = indices[n_validation:]

        return (X[train_indices], y[train_indices],
                X[val_indices], y[val_indices])


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        X_train, y_train, X_val, y_val = self._train_validation_split(X, y)

        self.init_ = self.loss_function.init_estimate(y_train)
        y_pred_train = np.full(len(y_train), self.init_)
        y_pred_val = np.full(len(y_val), self.init_) if y_val is not None else None

        self.train_score_ = []
        self.validation_score_ = []
        self.estimators_ = []

        best_val_score = np.inf
        no_improvement_count = 0

        for i in range(self.n_estimators):
            residuals = self.loss_function.negative_gradient(y_train, y_pred_train)

            X_sub, residuals_sub = self._subsample(X_train, residuals)
            tree = self._create_estimator()
            tree.fit(X_sub, residuals_sub)

            self.estimators_.append(tree)

            tree_pred_train = tree.predict(X_train)
            y_pred_train += self.learning_rate * tree_pred_train

            if y_val is not None:
                tree_pred_val = tree.predict(X_val)
                y_pred_val += self.learning_rate * tree_pred_val

            train_score = self.loss_function.loss(y_train, y_pred_train)
            self.train_score_.append(train_score)

            if y_val is not None:
                val_score = self.loss_function.loss(y_val, y_pred_val)
                self.validation_score_.append(val_score)

                # Early stopping check
                if self.n_iter_no_change is not None:
                    if val_score < best_val_score - self.tol:
                        best_val_score = val_score
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at iteration {i + 1}")
                        break

            if self.verbose and (i + 1) % 10 == 0:
                val_info = f", Val: {self.validation_score_[-1]:.6f}" if (y_val is not
                                                                          None) else ""
                print(f"Iter {i + 1:3d}: Train: {train_score:.6f}{val_info}")

        self._n_iter = len(self.estimators_)
        self._compute_feature_importances()

        return self


    def _compute_feature_importances(self) -> None:
        if not self.estimators_:
            return

        importances = np.zeros(self.n_features_)

        for tree in self.estimators_:
            tree_importances = tree._get_feature_importances(tree.root)
            importances += tree_importances

        if len(self.estimators_) > 0:
            importances /= len(self.estimators_)

        total_importance = np.sum(importances)
        if total_importance > 0:
            importances /= total_importance

        self.feature_importances_ = importances


    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Model not fitted yet!")

        # Start with initial prediction
        y_pred = np.full(X.shape[0], self.init_)

        # Add predictions from all trees
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred


    def staged_predict(self, X: np.ndarray) -> Generator[
        ndarray[tuple[int, ...], dtype[generic | Any]] | ndarray[
            tuple[int, ...], dtype[Any]] | Any, Any, None]:
        if not self.estimators_:
            raise ValueError("Model not fitted yet!")

        y_pred = np.full(X.shape[0], self.init_)

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
            yield y_pred.copy()


    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_decrease': self.min_impurity_decrease,
            'subsample': self.subsample,
            'max_features': self.max_features,
            'loss': self.loss,
            'random_state': self.random_state,
            'n_iter_': self._n_iter
        }
