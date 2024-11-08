from enum import Enum

from pydantic import BaseModel


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


"""
Enumeration that has two fields: numerical and categorical. 
"""


class Feature(BaseModel):
    name: str
    type: FeatureType

    def __str__(self) -> str:
        return "Feature: " + self.name + " of type " + self.type
