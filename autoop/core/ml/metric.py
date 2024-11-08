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
    """Base class for all metrics.
    """

    @abstractmethod
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        abstract method
        """
        pass


def get_metric(name: str) -> Metric:
    """Factory function to get a metric by name.
    :return: a metric instance given its str name
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
    MSE, regression
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate MSE
        """
        return float(np.square(np.subtract(y_true, y_pred)).mean())


class Accuracy(Metric):
    """
    Accuracy, classification
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate Accuracy
        """
        return float(np.sum(np.equal(y_true, y_pred)) / len(y_true))


class Recall(Metric):
    """
    Recall, classification
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate recall
        """
        unique_classes = np.unique(y_true)
        recall_scores = []

        for _class in unique_classes:
            true_positives = np.sum((y_pred == _class) & (y_true == _class))
            false_negatives = np.sum((y_pred != _class) & (y_true == _class))

            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0
            recall_scores.append(recall)

        average_recall = float(np.mean(recall_scores))
        return average_recall


class MeanAbsoluteError(Metric):
    """
    MAE, regression
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate MAE
        """
        return float(np.mean(np.abs(y_true - y_pred)))


class RootMeanSquaredError(Metric):
    """
    RMSE, regression
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate RMSE
        """
        mean_squared_error = np.square(np.subtract(y_true, y_pred)).mean()
        return float(np.sqrt(mean_squared_error))


class CoefficientOfDetermination(Metric):
    """
    Coefficient Of Determination, regression
    """
    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate Coefficient Of Determination
        """
        rss = np.sum(np.square(y_true - y_pred))
        tss = np.sum(np.square(y_true - np.mean(y_true)))
        coef_of_determination = 1 - (rss / tss)
        return float(coef_of_determination)


class CategoricalCrossEntropy(Metric):
    """
    Categorical Cross Entropy Loss, classification
    """

    def __call__(self, y_true: Union[List[float], np.ndarray],
                 y_pred: Union[List[float], np.ndarray]) -> float:
        """
        calculate Categorical Cross Entropy Loss
        """
        return float(-np.sum(y_true * np.log(y_pred)))


class ConfusionMatrix(Metric):
    """
    Confusion Matrix, classification
    """
    
    def __call__(self, y_true: Union[List[int], np.ndarray],
                 y_pred: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        calculate Confusion Matrix
        """
        max_label = max(max(y_true), max(y_pred))
        confusion_matrix = (
            np.zeros((max_label + 1,
                      max_label + 1), dtype=np.int16))

        for true, pred in zip(y_true, y_pred):
            confusion_matrix[true, pred] += 1

        return confusion_matrix
