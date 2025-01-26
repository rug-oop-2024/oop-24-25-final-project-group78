from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Custom exception raised when a specified path cannot be found.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the NotFoundError with a specific path.

        Args:
            path (str): Path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class for defining a storage functionality
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Abstract
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Abstract
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Abstract
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        Abstract
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Local file storage implementation
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initialize the LocalStorage with a base directory path.
        Args:
            base_path (str): The base directory for storage.
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data
        Args:
            data (bytes): Data to save.
            key (str): Relative path and filename under base path to save data.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data
        Args:
            key (str): Relative path and filename under base path to load data.
        Returns:
            bytes: The loaded data.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete
        Args:
            key (str):Path where data will be deleted.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all file paths
        Args:
            prefix (str): Directory prefix to start the listing from.
        Returns:
            List[str]: List of relative file paths under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Check if a path exists and raise NotFoundError if it does not.
        Args:
            path (str): Path to verify existence.
        Raises:
            NotFoundError: If the specified path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with the specified
        relative path to ensure OS compatibility.
        Args:
            path (str): Relative path to join with the base path.
        Returns:
            str: Full path.
        """
        return os.path.join(self._base_path, path)
