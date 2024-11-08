from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import Ridge as SKRidge

from autoop.core.ml.model.model import FacadeModel


class RegularizedRegression(FacadeModel, ABC):
    """

    """
    @abstractmethod
    def _initialize_model(self) -> Any:
        pass

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)
        self._type = "regression"
        self.params = params if (params
                                 is not None) \
            else {"alpha": 1, "max_iter": 1000}

    def validate_params(self, params: Dict) -> None:
        if "alpha" not in params.keys() or "max_iter" not in params.keys():
            raise ValueError("Invalid params. "
                             "They should contain alpha and max_iter.")

    def _set_params_from_model(self) -> None:
        self._params["coef_"] = self._wrapped_model.coef_
        self._params["intercept_"] = self._wrapped_model.intercept_


class Lasso(RegularizedRegression):
    def _initialize_model(self) -> SKLasso:
        return SKLasso(alpha=self._params["alpha"],
                       max_iter=self._params["max_iter"])


class Ridge(RegularizedRegression):
    def _initialize_model(self) -> SKRidge:
        return SKRidge(alpha=self._params["alpha"],
                       max_iter=self._params["max_iter"])


class MultipleLinearRegression(FacadeModel):
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)
        self._type = "regression"

    def _set_params_from_model(self) -> None:
        self._params["coef_"] = self._wrapped_model.coef_

    def validate_params(self, params: Dict) -> None:
        """
        Linear regression has no regularization parameters,
        so no specific validation is needed
        """
        pass

    def _initialize_model(self) -> SKLinearRegression:
        return SKLinearRegression()
