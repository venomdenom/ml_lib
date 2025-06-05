from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from regression_models.tree.criterions import ImpurityCriterion


@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    n_samples: int = 0
    impurity: float = 0.0


class DecisionTreeRegressor:

    def __init__(
            self,
            criterion: ImpurityCriterion,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            min_impurity_decrease: float = 0.0,
            max_features: Optional[int] = None,
            random_state: Optional[int] = None
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

        self.root = None
        self.n_features_ = None

        if random_state is not None:
            np.random.seed(random_state)


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)

        return self


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape

        node = Node(
            n_samples=n_samples,
            impurity=self.criterion.calculate(y)
        )

        if (self.max_depth is not None and depth >= self.max_depth) or \
                n_samples < self.min_samples_split or \
                n_samples < 2 * self.min_samples_leaf or \
                node.impurity == 0:
            node.value = float(np.mean(y))
            return node

        best_split = self._find_best_split(X, y)
        if best_split is None or best_split['improvement'] < self.min_impurity_decrease:
            node.value = float(np.mean(y))
            return node

        node.feature_idx = best_split['feature_idx']
        node.threshold = best_split['threshold']

        left_mask = X[:, best_split['feature_idx']] <= best_split['threshold']
        right_mask = ~left_mask

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node


    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[
        Dict[str, Any]]:
        n_samples, n_features = X.shape

        if self.max_features is None:
            features_to_consider = list(range(n_features))
        else:
            features_to_consider = np.random.choice(
                n_features,
                size=min(self.max_features, n_features),
                replace=False
            )

        best_split = None
        best_improvement = 0

        for feature_idx in features_to_consider:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            if len(unique_values) <= 1:
                continue

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or \
                        np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                improvement = self.criterion.improvement(y, y_left, y_right)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'improvement': improvement
                    }

        return best_split


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError('DecisionTreeRegressor must be fit before prediction')

        predictions = []
        for sample in X:
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)

        return np.array(predictions)


    def _predict_sample(self, sample: np.ndarray, node: Node) -> float:
        if node.value is not None:
            return node.value

        if sample[node.feature_idx] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)


    def get_depth(self) -> int:
        if self.root is None:
            return 0
        return self._get_depth(self.root)


    def _get_depth(self, node: Node) -> int:
        if node.value is not None:  # Лист
            return 1

        left_depth = self._get_depth(node.left) if node.left else 0
        right_depth = self._get_depth(node.right) if node.right else 0

        return 1 + max(left_depth, right_depth)


    def _get_feature_importances(self, node: Node) -> np.ndarray:
        importances = np.zeros(self.n_features_)


        def _traverse(node: Node):
            if node.value is not None:
                return

            importance = node.n_samples * node.impurity
            if node.left and node.right:
                left_importance = node.left.n_samples * node.left.impurity
                right_importance = node.right.n_samples * node.right.impurity
                improvement = importance - left_importance - right_importance
                importances[node.feature_idx] += improvement

            if node.left:
                _traverse(node.left)
            if node.right:
                _traverse(node.right)


        _traverse(node)
        return importances
