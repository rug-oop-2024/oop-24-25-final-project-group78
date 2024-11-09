from typing import List, Dict
from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A Dataset class inheriting from Artifact.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Dataset instance with the type set as "dataset".
        Args:
            *args, **kwargs: Positional and keyword
            arguments passed to the Artifact initializer.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0", tags: List = [],
                       metadata: Dict = {}) -> 'Dataset':
        """
        Args:
            data (pd.DataFrame): The data to be stored in the Dataset.
            name (str): Name of the dataset.
            asset_path (str): Path to the dataset asset.
            version (str, optional): Version of the dataset.
            tags (List, optional): Tags associated with the dataset.
            Defaults to an empty list.
            metadata (Dict, optional): Metadata for the dataset.
            Defaults to an empty dictionary.

        Returns:
            Dataset: A Dataset instance containing the DataFrame as CSV bytes.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            tags=tags,
            metadata=metadata
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset
        Returns:
            pd.DataFrame: The dataset as a DataFrame.
        """
        bytes_data = super().read()
        csv_str = bytes_data.decode()
        return pd.read_csv(io.StringIO(csv_str))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves the provided DataFrame
        Args:
            data (pd.DataFrame): The DataFrame to be saved.
        Returns:
            bytes: The CSV bytes stored in the parent Artifact.
        """
        bytes_data = data.to_csv(index=False).encode()
        return super().save(bytes_data)
