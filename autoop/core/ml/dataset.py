from typing import List, Dict

from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        """
        constructor for Dataset
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str,
                       version: str = "1.0.0",
                       tags: List = [],
                       metadata: Dict = {}) -> 'Dataset':
        """
        In the initialization of the object Dataset that
        it is returned in the static method, I added the arguments of the
        Artifact, that I previously mentioned.
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
        read the dataset
        return: pandas DataFrame
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        save pandas DataFrame as bytes and pass it 
        to the parent class's "save" method
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
