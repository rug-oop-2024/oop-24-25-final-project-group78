"""
classification and regression models
"""
from typing import Dict

from autoop.core.ml.model.model import Model

from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
    Lasso,
    Ridge,
)
from autoop.core.ml.model.classification.classification import (
    KNN,
    LogisticRegressionModel,
    SVCModel,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "Lasso",
    "Ridge"
]

CLASSIFICATION_MODELS = [
    "KNN",
    "LogisticRegressionModel",
    "SVCModel"
]


def get_model(model_name: str, params: Dict = None) -> Model:
    """Factory function to get a model by name."""
    if model_name in REGRESSION_MODELS:
        if model_name == "MultipleLinearRegression":
            return MultipleLinearRegression(params=params)
        elif model_name == "Lasso":
            return Lasso(params=params)
        elif model_name == "Ridge":
            return Ridge(params=params)
    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "KNN":
            return KNN(params=params)
        elif model_name == "LogisticRegressionModel":
            return LogisticRegressionModel(params=params)
        elif model_name == "SVCModel":
            return SVCModel(params=params)
    else:
        raise ValueError(
            f"Model '{model_name}' is not recognized. "
            f"Available models: {REGRESSION_MODELS + CLASSIFICATION_MODELS}")
