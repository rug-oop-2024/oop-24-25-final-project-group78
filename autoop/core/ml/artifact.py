from typing import Optional, List, Dict
from pydantic import BaseModel, Field, PrivateAttr
import base64
from copy import deepcopy


class Artifact(BaseModel):
    """
    Artifact model representing a data artifact with associated metadata,
    tags, and type.
    """
    asset_path: str
    name: str
    version: str
    data: Optional[bytes] = Field(default=None)
    _tags: List = PrivateAttr(default_factory=list)
    _metadata: Dict = PrivateAttr(default_factory=dict)
    type: str

    def get_id(self) -> str:
        """
        Generates a unique ID for the artifact
        based on its asset path and version.
        Returns:
            str: Base64 encoded asset path concatenated with the version.
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Reads and returns the data of the artifact.
        Returns:
            bytes: The data of the artifact.
        Raises:
            ValueError: If the data is None.
        """
        if self.data is None:
            raise ValueError("The data is None.")
        else:
            return self.data

    def save(self, data: bytes) -> bytes:
        """
        Saves the provided data
        Args:
            data (bytes): Data to be saved in the artifact.
        Returns:
            bytes: The saved data.
        """
        self.data = data
        return self.data

    @property
    def metadata(self) -> Dict:
        """
        Retrieves a copy of the metadata associated with the artifact.
        Returns:
            Dict: A deepcopy of the metadata dictionary.
        """
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        """
        Sets the metadata for the artifact.
        Args:
            metadata (Dict): Dictionary containing metadata.
        Raises:
            TypeError: If metadata is not a dictionary.
        """
        if not isinstance(metadata, Dict):
            raise TypeError("Wrong type.")
        self._metadata = metadata

    @property
    def tags(self) -> List:
        """
        Retrieves a copy of the tags associated with the artifact.
        Returns:
            List: A deepcopy of the tags list.
        """
        return deepcopy(self._tags)

    @tags.setter
    def tags(self, tags: List) -> None:
        """
        Sets the tags for the artifact.
        Args:
            tags (List): List of tags.
        Raises:
            TypeError: If tags is not a list.
        """
        if not isinstance(tags, List):
            raise TypeError("Wrong type.")
        self._tags = tags

    @property
    def id(self) -> str:
        """
        Retrieves the unique ID of the artifact.
        Returns:
            str: The unique ID generated by the get_id method.
        """
        return self.get_id()
