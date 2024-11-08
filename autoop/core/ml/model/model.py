import pickle
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Dict, Any


class Model(ABC):
    _params: Dict
    _type: str
    """
    encapsulation
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)
    """
    Put the accent on the fact that the models
    are functions and they are callable. It returns what the
    "predict" method returns on that x.
    """

    @abstractmethod
    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @property
    def params(self) -> Dict:
        return deepcopy(self._params)

    @params.setter
    def params(self, params: Dict) -> None:
        self.validate_params(params)
        self._params = params

    @abstractmethod
    def validate_params(self, params: Dict) -> None:
        pass

    @property
    def type(self) -> str:
        return self._type

    def to_artifact(self, name: str) -> Artifact:
        return Artifact(name=name, data=pickle.dump(self._params))
    """
    Returns an artifact with a name given and the data being parameters.
    """


class FacadeModel(Model, ABC):
    """

    """
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = params if params is not None else {}
        self._wrapped_model = self._initialize_model()

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        self._wrapped_model.fit(train_x, train_y)
        self._set_params_from_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._wrapped_model.predict(x)

    @abstractmethod
    def _initialize_model(self) -> Any:
        pass

    @abstractmethod
    def _set_params_from_model(self) -> None:
        pass
