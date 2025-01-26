from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.metric import MeanSquaredError


class TestPipeline(unittest.TestCase):
    """
    Unit tests for the Pipeline class
    """

    def setUp(self) -> None:
        """
        Set up the test environment
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(filter(lambda x: x.name != "age",
                                       self.features)),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        """
        Test the initialization of the Pipeline instance
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        """
        Test the feature preprocessing step
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        """
        Test the data splitting functionality
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(self.pipeline._train_X[0].shape[0],
                         int(0.8 * self.ds_size))
        self.assertEqual(self.pipeline._test_X[0].shape[0],
                         self.ds_size - int(0.8 * self.ds_size))

    def test_train(self):
        """
        Test the training process of the model within the Pipeline
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.params)

    def test_evaluate(self):
        """
        Test the evaluation process of the Pipeline
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results_test)
        self.assertEqual(len(self.pipeline._metrics_results_test), 1)


if __name__ == "__main__":
    unittest.main()
