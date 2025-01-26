from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """Manages the registration and storage of artifacts"""

    def __init__(self, database: Database, storage: Storage) -> None:
        """Initializes the ArtifactRegistry with database and storage.

        Args:
            database (Database): The database to store artifact metadata.
            storage (Storage): The storage to save artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """Registers an artifact by saving its
        data in storage and metadata in the database.
        Args:
            artifact (Artifact): The artifact to register.
        """
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """Lists all artifacts or filters by type if specified.
        Args:
            type (str, optional): The type of
            artifacts to list. Defaults to None.
        Returns:
            List[Artifact]: A list of artifacts
            matching the specified type (if given).
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type_=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Retrieves an artifact based on its ID.
        Args:
            artifact_id (str): The ID of the artifact to retrieve.
        Returns:
            Artifact: The artifact corresponding to the given ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type_=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """Deletes an artifact by its ID from both storage and database.
        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """Manages the AutoML system"""

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """Initializes the AutoML system with specified storage and database.
        Args:
            storage (LocalStorage): Storage instance for saving artifacts.
            database (Database): Database
            instance for saving artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """Singleton access method to get
        or create an instance of AutoMLSystem.
        Returns:
            AutoMLSystem: The singleton instance of the AutoMLSystem.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """Provides access to the artifact registry.
        Returns:
            ArtifactRegistry: The registry for managing artifacts.
        """
        return self._registry
