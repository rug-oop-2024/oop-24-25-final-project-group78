import pickle
from abc import abstractmethod, ABC

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict


class Model(ABC):
    """
    Base class for machine learning models
    """
    _params: Dict  # Model parameters
    _type: str  # Model type (e.g., "regression" or "classification")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Enables the model to be called as a function
        Args:
            x (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted values for the input data.
        """
        return self.predict(x)

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Abstract method for fitting the model
        Args:
            train_x (np.ndarray): Training data features.
            train_y (np.ndarray): Training data labels/targets.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Abstract method for predicting outputs
        Args:
            x (np.ndarray): Input data for which predictions are required.
        Returns:
            np.ndarray: Predictions for the input data.
        """
        pass

    @property
    def params(self) -> Dict:
        """
        Gets a deepcopy of the model parameters to ensure encapsulation.
        Returns:
            Dict: A deepcopy of the model's parameters.
        """
        return deepcopy(self._params)

    @params.setter
    def params(self, params: Dict) -> None:
        """
        Sets the model parameters after validation.
        Args:
            params (Dict): Dictionary of model parameters to set.
        Raises:
            ValueError: If parameters do not meet validation criteria.
        """
        self.validate_params(params)
        self._params = params

    @abstractmethod
    def validate_params(self, params: Dict) -> None:
        """
        Abstract method for validating model parameters.
        Args:
            params (Dict): Dictionary of parameters to validate.
        Raises:
            ValueError: If parameters are invalid for the model.
        """
        pass

    @property
    def type(self) -> str:
        """
        Returns the model type.
        Returns:
            str: The type of the model
            (e.g., "regression" or "classification").
        """
        return self._type

    def to_artifact(self, name: str) -> Artifact:
        """
        Creates an artifact containing the model's parameters.
        Args:
            name (str): The name of the artifact.
        Returns:
            Artifact: An Artifact instance
            containing the serialized parameters.
        """
        return Artifact(name=name, data=pickle.dumps(self._params))


class FacadeModel(Model, ABC):
    """
    Implements the Facade design pattern to simplify model interactions.
    """
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """
        Initializes the Facade model with
        parameters and sets up the wrapped model.
        Args:
            *args: Additional positional arguments for base initialization.
            params (Dict, optional): Initial parameters
            for the model. Defaults to an empty dictionary.
            **kwargs: Additional keyword arguments for base initialization.
        """
        super().__init__(*args, **kwargs)
        self.params = params if params is not None else {}
        self._wrapped_model = self._initialize_model()

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Fits the wrapped model using the provided training data.
        Args:
            train_x (np.ndarray): Training data features.
            train_y (np.ndarray): Training data labels/targets.
        """
        self._wrapped_model.fit(train_x, train_y)
        self._set_params_from_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts outputs for the input data using the wrapped model.
        Args:
            x (np.ndarray): Input data for which predictions are required.
        Returns:
            np.ndarray: Predictions for the input data.
        """
        return self._wrapped_model.predict(x)

    @abstractmethod
    def _initialize_model(self) -> (
            Ridge 
            | Lasso 
            | LinearRegression 
            | LogisticRegression 
            | KNeighborsClassifier
            | SVC
    ):
        """
        Abstract method for initializing the wrapped model instance.
        Returns:
            Any: The initialized model instance.
        """
        pass

    @abstractmethod
    def _set_params_from_model(self) -> None:
        """
        Abstract method for updating the facade
        model's parameters from the wrapped model.
        """
        pass
