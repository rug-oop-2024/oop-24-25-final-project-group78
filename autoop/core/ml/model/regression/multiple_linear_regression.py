from abc import abstractmethod
from typing import Dict
from typing import Union

from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import Ridge as SKRidge

from autoop.core.ml.model.model import FacadeModel


class RegularizedRegression(FacadeModel):
    """Base class containing common functionality
    for regularized regression models."""

    @abstractmethod
    def _initialize_model(self) -> Union[SKLasso, SKRidge]:
        """
        Initializes and returns the specific regression model.
        Note: I could not use a super class of both Ridge and Lasso
        from sklearn, like LinearModel, because was in a private
        package(for a return type hint).
        Returns:
            Union[SKLasso, SKRidge]
        """
        pass

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """Constructs a RegularizedRegression instance.
        Args:
            *args: Positional arguments for the parent class.
            params (Dict, optional): Dictionary containing model parameters.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, params=params, **kwargs)
        self._type = "regression"
        self.params = params

    def _validate_params(self, params: Dict) -> None:
        """Validates the provided parameters to
        ensure they include "alpha" and "max_iter".
        Args:
            params (Dict): Dictionary of parameters to validate.
        Raises:
            ValueError: If required keys "alpha" and "max_iter" are missing.
        """
        if "alpha" not in params.keys() or "max_iter" not in params.keys():
            raise ValueError("Invalid params. They should "
                             "contain 'alpha' and 'max_iter'."
                             f" {params} were given.")

    def _set_default_params(self) -> None:
        """
        Sets the default parameters for the specific model.
        """
        self._params = {"alpha": 1, "max_iter": 1000}

    def _set_params_from_model(self) -> None:
        """Saves the coefficients and intercept of the regression model"""
        self._params["coef_"] = self._wrapped_model.coef_
        self._params["intercept_"] = self._wrapped_model.intercept_


class Lasso(RegularizedRegression):
    """Lasso regression model
    """

    def _initialize_model(self) -> SKLasso:
        """Initializes and returns an instance of the Lasso regression model.
        Returns:
            SKLasso: Lasso model from sklearn.
        """
        return SKLasso(alpha=self._params["alpha"],
                       max_iter=self._params["max_iter"])


class Ridge(RegularizedRegression):
    """Ridge regression model
    """

    def _initialize_model(self) -> SKRidge:
        """Initializes and returns an instance of the Ridge regression model.
        Returns:
            SKRidge: Ridge model from sklearn.
        """
        return SKRidge(alpha=self._params["alpha"],
                       max_iter=self._params["max_iter"])


class MultipleLinearRegression(FacadeModel):
    """Multiple Linear Regression model"""

    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        """Constructs a MultipleLinearRegression
        instance with type set to "regression".
        Args:
            *args: Positional arguments for the parent class.
            params (Dict, optional): Dictionary containing model parameters.
            **kwargs: Keyword arguments for the parent class.
        """
        super().__init__(*args, params=params, **kwargs)
        self._type = "regression"

    def _set_params_from_model(self) -> None:
        """Saves the coefficients and intercept
        of the linear regression model"""
        self._params["coef_"] = self._wrapped_model.coef_
        self._params["intercept_"] = self._wrapped_model.intercept_

    def _validate_params(self, params: Dict) -> None:
        """No specific validation is required for linear
        regression as it has no regularization parameters."""
        pass

    def _initialize_model(self) -> SKLinearRegression:
        """Initializes and returns an instance of the Linear Regression model.
        Returns:
            SKLinearRegression: LinearRegression model from sklearn.
        """
        return SKLinearRegression()

    def _set_default_params(self) -> None:
        """
        Sets the default parameters for the specific model.
        """
        self._params = {}
