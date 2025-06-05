from abc import ABC, abstractmethod

import numpy as np


class ImpurityCriterion(ABC):

    @abstractmethod
    def calculate(self, y: np.ndarray) -> float:
        pass


    @abstractmethod
    def improvement(
            self,
            y_parent: np.ndarray,
            y_left: np.ndarray,
            y_right: np.ndarray
    ) -> float:
        pass


class MSECriterion(ImpurityCriterion):

    @classmethod
    def calculate(cls, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0

        mean_y = np.mean(y)
        return float(np.mean((y - mean_y) ** 2))


    @classmethod
    def improvement(
            cls,
            y_parent: np.ndarray,
            y_left: np.ndarray,
            y_right: np.ndarray
    ) -> float:
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        impurity_parent = cls.calculate(y_parent)
        impurity_left = cls.calculate(y_left)
        impurity_right = cls.calculate(y_right)

        weighted_impurity = (n_left / n_parent) * impurity_left + (
                n_right / n_parent) * impurity_right

        return impurity_parent - weighted_impurity


class MAECriterion(ImpurityCriterion):

    @classmethod
    def calculate(cls, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0

        median_y = np.median(y)
        return np.mean(np.abs(y - median_y))


    @classmethod
    def improvement(
            cls,
            y_parent: np.ndarray,
            y_left: np.ndarray,
            y_right: np.ndarray
    ) -> float:
        if len(y_left) == 0 or len(y_right) == 0:
            return 0

        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        impurity_parent = cls.calculate(y_parent)
        impurity_left = cls.calculate(y_left)
        impurity_right = cls.calculate(y_right)

        weighted_impurity = (n_left / n_parent) * impurity_left + (
                n_right / n_parent) * impurity_right

        return impurity_parent - weighted_impurity
