from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from typing import Dict, Any


from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from autoop.core.ml.model.model import FacadeModel


class KNN(FacadeModel):
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        self._params["n_neighbors"] = self._wrapped_model.n_neighbors
        self._params["weights"] = self._wrapped_model.weights
        self._params["algorithm"] = self._wrapped_model.algorithm
        self._params["leaf_size"] = self._wrapped_model.leaf_size
        self._params["metric"] = self._wrapped_model.metric
        self._params["p"] = self._wrapped_model.p

    def validate_params(self, params: Dict) -> None:
        if "n_neighbors" not in params.keys() or "weights" not in params.keys():
            raise ValueError("Params need to include \"n_neighbors\" and \"weights\".")

    def _initialize_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=self._params["n_neighbors"], weights=self._params["weights"])


class LogisticRegressionModel(FacadeModel):
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        self._params["C"] = self._wrapped_model.C
        self._params["max_iter"] = self._wrapped_model.max_iter

    def validate_params(self, params: Dict) -> None:
        if "C" not in params.keys() or "max_iter" not in params.keys():
            raise ValueError("Params need to include \"C\" and \"max_iter\".")

    def _initialize_model(self) -> SKLogisticRegression:
        return SKLogisticRegression(C=self._params["C"], max_iter=self._params["max_iter"])


class SVCModel(FacadeModel):
    def __init__(self, *args, params: Dict = None, **kwargs) -> None:
        super().__init__(*args, params=params, **kwargs)
        self._type = "classification"

    def _set_params_from_model(self) -> None:
        self._params["C"] = self._wrapped_model.C
        self._params["kernel"] = self._wrapped_model.kernel
        self._params["gamma"] = self._wrapped_model.gamma

    def validate_params(self, params: Dict) -> None:
        required_params = ["C", "kernel", "gamma"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Params need to include \"{param}\".")

    def _initialize_model(self) -> SVC:
        return SVC(C=self._params["C"], kernel=self._params["kernel"], gamma=self._params["gamma"])
