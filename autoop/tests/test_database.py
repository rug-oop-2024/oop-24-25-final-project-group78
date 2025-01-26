import unittest
from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile


class TestDatabase(unittest.TestCase):
    """
    Unit tests for the Database class
    """

    def setUp(self):
        """
        Set up the test environment
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        """
        Test the initialization of the Database instance
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self):
        """
        Test the 'set' functionality of the Database
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self):
        """
        Test the 'delete' functionality of the Database
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistence(self):
        """
        Test the persistence of data in the Database
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self):
        """
        Test the 'refresh' functionality
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        """
        Test the 'list' functionality
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # Check that the collection contains the key-value pair
        self.assertIn((key, value), self.db.list("collection"))


if __name__ == "__main__":
    unittest.main()
