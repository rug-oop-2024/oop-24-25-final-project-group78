from enum import Enum
from pydantic import BaseModel


class FeatureType(str, Enum):
    """
    Enumeration representing the possible types of a feature:
    numerical and categorical.
    """
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class Feature(BaseModel):
    """
    Feature model representing a data feature with a name and type.
    """
    name: str
    type: FeatureType

    def __str__(self) -> str:
        """

        Returns:
            str: A string describing the feature with its name and type.
        """
        return "Feature: " + self.name + " of type " + self.type
