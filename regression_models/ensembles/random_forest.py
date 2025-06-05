from typing import Optional, Union, Dict, Any

import numpy as np

from regression_models.tree.criterions import ImpurityCriterion, MSECriterion
from regression_models.tree.decision_tree import DecisionTreeRegressor


class RandomForestRegressor:
    def __init__(
            self,
            n_estimators: int,
            criterion: ImpurityCriterion = MSECriterion,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_impurity_decrease: float = 0.0,
            max_features: Union[str, int, float] = 'sqrt',
            bootstrap: bool = True,
            oob_score: bool = False,
            n_jobs: Optional[int] = None,
            random_state: Optional[int] = None,
            verbose: bool = False
    ) -> None:
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None

        if random_state is not None:
            np.random.seed(random_state)


    def _get_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, str):
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


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        max_features = self._get_max_features(n_features)

        self.estimators_ = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"Training tree {i + 1}/{self.n_estimators}")

            oob_indices = []
            if self.bootstrap:
                bootstrap_indices = np.random.choice(
                    n_samples, size=n_samples, replace=True
                )
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                # OOB indices
                if self.oob_score:
                    oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            else:
                X_bootstrap = X.copy()
                y_bootstrap = y.copy()

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=max_features,
                random_state=None
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

        if self.oob_score:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                oob_predictions[valid_oob] /= oob_counts[valid_oob]
                self.oob_score_ = 1 - np.mean(
                    (y[valid_oob] - oob_predictions[valid_oob]) ** 2
                ) / np.var(y[valid_oob])
            else:
                self.oob_score_ = None

        self._compute_feature_importance()

        return self


    def _compute_feature_importance(self):
        importances = np.zeros(self.n_features_)

        for tree in self.estimators_:
            tree_importances = tree._get_feature_importances(tree.root)
            importances += tree_importances

        importances /= len(self.estimators_)
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances /= total_importance

        self.feature_importances_ = importances


    def _base_predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise ValueError("Random Forest not fitted yet!")

        predictions = np.zeros((X.shape[0], len(self.estimators_)))

        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)

        return predictions


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(self._base_predict(X), axis=1)


    def predict_std(self, X: np.ndarray) -> np.ndarray:
        return np.std(self._base_predict(X), axis=1)


    def get_model_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state
        }
