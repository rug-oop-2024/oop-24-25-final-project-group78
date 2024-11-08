from typing import List

import pandas as pd

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature, FeatureType


def _get_feature_type(df: pd.DataFrame, column_name: object) -> FeatureType:
    """Private method (encapsulation) that takes as an argument a data frame and the column name. It accesses the
    column of the data frame by indexing the data frame on that column and then checks weather the type is categorical 
    or numerical and returns the associated feature type value.
    Args:
        df: data frame
        column_name: the name of the column
    Returns:
        FeatureType: Feature type associated to that feature.
    """
    column = df[column_name]
    if pd.api.types.is_numeric_dtype(column):
        return FeatureType.NUMERICAL
    if df[column_name].dtype.name == "object":
        return FeatureType.CATEGORICAL
    raise ValueError("Unknown feature type")


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Reads the dataset as a dataframe using the method read, accesses the columns of the dataframe and for each column
    of the dataframe, gets the feature type using the private method "get_feature_type" (encapsulation reasons), creates
    a feature with the name of the column and the feature type, appends it to the list of the features and returns the
    features.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    data_frame = dataset.read()
    columns = data_frame.columns
    for column in columns:
        feature_type = _get_feature_type(data_frame, column)
        feature = Feature(name=column, type=feature_type)
        features.append(feature)
    return features
