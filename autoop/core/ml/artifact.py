from typing import Optional, List, Dict

from pydantic import BaseModel, Field, PrivateAttr
import base64
from copy import deepcopy


class Artifact(BaseModel):
    asset_path: str
    name: str
    version: str
    data: Optional[bytes] = Field(default=None)
    _tags: List = PrivateAttr(default_factory=list)
    _metadata: Dict = PrivateAttr(default_factory=dict)
    type: str
    """
    I added the necessary fields that were
    mentioned in Pipeline (tags/ metadata/ type).
    """

    def get_id(self) -> str:
        """
        :return: The id formatted as given in the instructions
        """
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Returns the data of the Artifact if it is not None.
        """
        if self.data is None:
            raise ValueError("The data is None.")
        else:
            return self.data

    def save(self, data: bytes) -> bytes:
        """
        Save the given data in "self.data" and returns the data.
        """
        self.data = data
        return self.data

    @property
    def metadata(self) -> Dict:
        return deepcopy(self._metadata)

    @metadata.setter
    def metadata(self, metadata: Dict) -> None:
        if not isinstance(metadata, Dict):
            raise TypeError("Wrong type.")
        self._metadata = metadata

    @property
    def tags(self) -> List:
        return deepcopy(self._tags)

    @tags.setter
    def tags(self, tags: List) -> None:
        if not isinstance(tags, List):
            raise TypeError("Wrong type.")
        self._tags = tags

    @property
    def id(self) -> str:
        return self.get_id()
