import pickle
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict, Any


class Model(ABC):
    """
    base class for the model
    """
    _params: Dict
    _type: str
    """
    encapsulation
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Put the accent on the fact that the models
        are functions, and they are callable. It returns what the
        "predict" method returns on that x.
        """
        return self.predict(x)

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        abstract method for fit 
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        abstract method for predict
        """
        pass

    @property
    def params(self) -> Dict:
        """
        get a safe copy of the parameters 
        """
        return deepcopy(self._params)

    @params.setter
    def params(self, params: Dict) -> None:
        """
        set the parameters 
        """
        self.validate_params(params)
        self._params = params

    @abstractmethod
    def validate_params(self, params: Dict) -> None:
        """
        abstract method for validation
        """
        pass

    @property
    def type(self) -> str:
        """
        get the type
        """
        return self._type

    def to_artifact(self, name: str) -> Artifact:
        """
        Returns an artifact with a name given and the data being parameters.
        """
        return Artifact(name=name, data=pickle.dump(self._params))


class FacadeModel(Model, ABC):
    """
    Facade model
    """
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """
        initialize Facade model
        """
        super().__init__(*args, **kwargs)
        self.params = params if params is not None else {}
        self._wrapped_model = self._initialize_model()

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        fit model
        """
        self._wrapped_model.fit(train_x, train_y)
        self._set_params_from_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict using wrapped model
        """
        return self._wrapped_model.predict(x)

    @abstractmethod
    def _initialize_model(self) -> Any:
        """
        initialize the model
        """
        pass

    @abstractmethod
    def _set_params_from_model(self) -> None:
        """
        set the parameters
        """
        pass
