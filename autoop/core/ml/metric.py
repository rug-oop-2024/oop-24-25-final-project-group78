from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "recall",
    "mean_absolute_error",
    "root_mean_squared_error",
    "coefficient_of_determination",
    "cross_entropy",
    "confusion_matrix"
]


class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @abstractmethod
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Abstract method
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: Computed metric value.
        """
        pass


def get_metric(name: str) -> Metric:
    """
    Factory function to get a metric instance by name.
    Args:
        name (str): Name of the metric.
    Returns:
        Metric: An instance of the specified metric.
    Raises:
        ValueError: If the metric name is not recognized.
    """
    if name not in METRICS:
        raise ValueError("Unknown metric name")
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "recall":
        return Recall()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    elif name == "coefficient_of_determination":
        return CoefficientOfDetermination()
    elif name == "cross_entropy":
        return CategoricalCrossEntropy()
    elif name == "confusion_matrix":
        return ConfusionMatrix()


class MeanSquaredError(Metric):
    """
    Mean Squared Error for regression tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Mean Squared Error.
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The MSE value.
        """
        return float(np.square(np.subtract(y_true, y_pred)).mean())


class Accuracy(Metric):
    """
    Accuracy metric for classification tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate accuracy.
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The accuracy value.
        """
        return float(np.sum(np.equal(y_true, y_pred)) / len(y_true))


class Recall(Metric):
    """
    Recall metric for classification tasks
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate recall
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The recall value.
        """
        unique_classes = np.unique(y_true)
        recall_scores = []

        for _class in unique_classes:
            true_positives = np.sum((y_pred == _class) & (y_true == _class))
            false_negatives = np.sum((y_pred != _class) & (y_true == _class))

            recall = true_positives / (true_positives + false_negatives) \
                if (true_positives + false_negatives) > 0 else 0
            recall_scores.append(recall)

        return float(np.mean(recall_scores))


class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error for regression tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Mean Absolute Error.
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The MAE value.
        """
        return float(np.mean(np.abs(y_true - y_pred)))


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error for regression tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Root Mean Squared Error.
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The RMSE value.
        """
        mean_squared_error = np.square(np.subtract(y_true, y_pred)).mean()
        return float(np.sqrt(mean_squared_error))


class CoefficientOfDetermination(Metric):
    """
    Coefficient of Determination for regression tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Coefficient of Determination
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float: The COD value.
        """
        rss = np.sum(np.square(y_true - y_pred))
        tss = np.sum(np.square(y_true - np.mean(y_true)))
        return float(1 - (rss / tss))


class CategoricalCrossEntropy(Metric):
    """
    Categorical Cross Entropy Loss for classification tasks.
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        Calculate Categorical Cross Entropy Loss.
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth values.
            y_pred (Union[List[float], np.ndarray]): Predicted probabilities.
        Returns:
            float: The cross-entropy loss value.
        """
        return float(-np.sum(y_true * np.log(y_pred)))


class ConfusionMatrix(Metric):
    """
    Confusion Matrix for classification tasks.
    """
    def __call__(self, y_true: Union[List[int], np.ndarray],
                 y_pred: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Calculate Confusion Matrix.
        Args:
            y_true (Union[List[int], np.ndarray]):
            Ground truth values (integer labels).
            y_pred (Union[List[int], np.ndarray]):
            Predicted values (integer labels).
        Returns:
            np.ndarray: The confusion matrix
        """
        max_label = max(max(y_true), max(y_pred))
        confusion_matrix = (
            np.zeros((max_label + 1,
                      max_label + 1), dtype=np.int16))

        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true, pred] += 1

        return confusion_matrix
