from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

METRICS_CLASSIFICATION = [
    "accuracy",
    "recall",
    "macro_f1_score"
]


METRICS_REGRESSION = [
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_squared_error",
    "coefficient_of_determination",
]


class Metric(ABC):
    """
    Abstract base class for all metrics.
    """

    @abstractmethod
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float | np.ndarray:
        """
        Abstract method
        Args:
            y_true (Union[List[float], np.ndarray]): Ground truth.
            y_pred (Union[List[float], np.ndarray]): Predicted values.
        Returns:
            float | np.ndarray
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
    if name not in METRICS_REGRESSION and name not in METRICS_CLASSIFICATION:
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
    elif name == "macro_f1_score":
        return MacroF1Score()


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


class MacroF1Score(Metric):
    """
    Macro-Averaged F1 Score for multi-class classification.
    """
    def __call__(self, y_true: Union[List[int], np.ndarray],
                 y_pred: Union[List[int], np.ndarray]) -> float:
        """
        Calculate Macro-Averaged F1 Score.

        Args:
            y_true (Union[List[int], np.ndarray]): Ground truth values.
            y_pred (Union[List[int], np.ndarray]): Predicted values.

        Returns:
            float: The Macro F1 Score value.
        """

        if y_true.size == 0 or y_pred.size == 0:
            raise ValueError("y_true or y_pred is empty!")

        unique_classes = np.unique(y_true)
        f1_scores = []

        for _class in unique_classes:
            true_positive = np.sum((y_pred == _class) & (y_true == _class))
            false_positive = np.sum((y_pred == _class) & (y_true != _class))
            false_negative = np.sum((y_pred != _class) & (y_true == _class))

            precision = true_positive / (true_positive + false_positive) \
                if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) \
                if (true_positive + false_negative) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores)

        return float(macro_f1)
