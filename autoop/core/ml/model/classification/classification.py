from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from typing import Dict
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from autoop.core.ml.model.model import FacadeModel


class KNN(FacadeModel):
    """K-Nearest Neighbors (KNN) model for classification tasks."""

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """Initializes the KNN model with specified parameters.
        Args:
            *args: Positional arguments for the parent class.
            params (Dict, optional): Model parameters
            including 'n_neighbors' and 'weights'.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        """Extracts parameters from the trained KNN model and saves them."""
        self._params["n_neighbors"] = self._wrapped_model.n_neighbors
        self._params["weights"] = self._wrapped_model.weights
        self._params["algorithm"] = self._wrapped_model.algorithm
        self._params["leaf_size"] = self._wrapped_model.leaf_size
        self._params["metric"] = self._wrapped_model.metric
        self._params["p"] = self._wrapped_model.p

    def validate_params(self, params: Dict) -> None:
        """Validates that the required parameters
        Args:
            params (Dict): Dictionary of model parameters.
        Raises:
            ValueError: If 'n_neighbors' or 'weights' are not in `params`.
        """
        if "n_neighbors" not in params or "weights" not in params:
            raise ValueError('Params need to '
                             'include "n_neighbors" and "weights".')

    def _initialize_model(self) -> KNeighborsClassifier:
        """Initializes and returns the KNeighborsClassifier model.
        Returns:
            KNeighborsClassifier: An instance of
            KNeighborsClassifier with specified parameters.
        """
        return KNeighborsClassifier(n_neighbors=self._params["n_neighbors"],
                                    weights=self._params["weights"])


class LogisticRegressionModel(FacadeModel):
    """Logistic Regression model for classification tasks."""

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """Initializes the Logistic Regression model.
        Args:
            *args: Positional arguments for the parent class.
            params (Dict, optional): Model
            parameters including 'C' and 'max_iter'.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        """Extracts parameters from the trained
        Logistic Regression model and saves them"""
        self._params["C"] = self._wrapped_model.C
        self._params["max_iter"] = self._wrapped_model.max_iter

    def validate_params(self, params: Dict) -> None:
        """Validates that the required parameters.
        Args:
            params (Dict): Dictionary of model parameters.
        Raises:
            ValueError: If 'C' or 'max_iter' are not in `params`.
        """
        if "C" not in params or "max_iter" not in params:
            raise ValueError('Params need to include "C" and "max_iter".')

    def _initialize_model(self) -> SKLogisticRegression:
        """Initializes and returns the Logistic Regression model.
        Returns:
            SKLogisticRegression: An instance of
            LogisticRegression with specified parameters.
        """
        return SKLogisticRegression(C=self._params["C"],
                                    max_iter=self._params["max_iter"])


class SVCModel(FacadeModel):
    """Support Vector Classifier (SVC) model for classification tasks."""

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """Initializes the SVC model
        Args:
            *args: Positional arguments for the parent class.
            params (Dict, optional): Model parameters
            including 'C', 'kernel', and 'gamma'.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        """Extracts parameters from the trained SVC model and saves them"""
        self._params["C"] = self._wrapped_model.C
        self._params["kernel"] = self._wrapped_model.kernel
        self._params["gamma"] = self._wrapped_model.gamma

    def validate_params(self, params: Dict) -> None:
        """Validates that the required parameters
        Args:
            params (Dict): Dictionary of model parameters.
        Raises:
            ValueError: If 'C', 'kernel', or 'gamma' are not in `params`.
        """
        required_params = ["C", "kernel", "gamma"]
        for param in required_params:
            if param not in params:
                raise ValueError(f'Params need to include "{param}".')

    def _initialize_model(self) -> SVC:
        """Initializes and returns the SVC model.
        Returns:
            SVC: An instance of SVC with specified parameters.
        """
        return SVC(C=self._params["C"],
                   kernel=self._params["kernel"], gamma=self._params["gamma"])
