from typing import List, Dict, Any
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline:
    """
    Manages different stages of data processing and model training
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8) -> None:
        """
        Initializes the Pipeline with the necessary components.
        Args:
            metrics (List[Metric]): List of metrics to evaluate the model.
            dataset (Dataset): Dataset object containing the data.
            model (Model): The model to be trained and evaluated.
            input_features (List[Feature]): Features
            used as inputs for the model.
            target_feature (Feature): The target feature for the model.
            split (float, optional): Ratio for
            train-test split. Defaults to 0.8.
        Raises:
            ValueError: If model type does not match the target feature type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (target_feature.type == "categorical"
                and model.type != "classification"):
            raise ValueError("Model type must be "
                             "classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression "
                             "for continuous target feature")

    def __str__(self) -> str:
        """
        Provides a string representation
        Returns:
            str: String representation of the Pipeline.
        """
        return f"""
        Pipeline(
        model={self._model.type},
        input_features={list(map(str, self._input_features))},
        target_feature={str(self._target_feature)},
        split={self._split},
        metrics={list(map(str, self._metrics))},
                )
    """

    @property
    def model(self) -> Model:
        """
        Getter for the model used in the Pipeline.
        Returns:
            Model: The model used in the Pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves artifacts
        Returns:
            List[Artifact]: List of Artifact objects representing artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Dict) -> None:
        """
        Registers an artifact
        Args:
            name (str): The name of the artifact.
            artifact (Dict): The artifact data.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses features
        """
        (target_feature_name, target_data,
         artifact) = (
            preprocess_features([self._target_feature],
                                self._dataset))[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = (
            preprocess_features(self._input_features, self._dataset))
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = \
            [data for feature_name, data, artifact in input_results]

    def _split_data(self) -> None:
        """
        Splits the data into training and testing sets
        """
        split = self._split
        self._train_X = \
            [vector[:int(split
                         * len(vector))] for vector in self._input_vectors]
        self._test_X = \
            [vector[int(split
                        * len(vector)):] for vector in self._input_vectors]
        self._train_y = (
                            self._output_vector)[:int(split
                                                      * len(self._output_vector))]
        self._test_y = (
                           self._output_vector)[int(split
                                                    * len(self._output_vector)):]

    @staticmethod
    def _compact_vectors(vectors: List[np.array]) -> np.array:
        """
        Compacts a list of vectors
        Args:
            vectors (List[np.array]): List of arrays to concatenate.
        Returns:
            np.array: Concatenated array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = Pipeline._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model on both training
        and testing sets using specified metrics.
        """
        X = Pipeline._compact_vectors(self._test_X)
        Y = self._test_y

        X_train = Pipeline._compact_vectors(self._train_X)
        Y_train = self._train_y

        self._metrics_results_test = []
        self._metrics_results_train = []

        predictions = self._model.predict(X)
        predictions_train = self._model.predict(X_train)

        for metric in self._metrics:
            result_test, result_train = (metric(predictions, Y),
                                         metric(predictions_train, Y_train))
            self._metrics_results_test.append((metric, result_test))
            self._metrics_results_train.append((metric, result_train))
        self._predictions = predictions

    def execute(self) -> Dict[str, Any]:
        """
        Executes the full pipeline
        Returns:
            dict: Dictionary containing metric results for both training
                  and testing sets, and model predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics on training set": self._metrics_results_train,
            "metrics on evaluation set": self._metrics_results_test,
            "predictions": self._predictions,
        }
