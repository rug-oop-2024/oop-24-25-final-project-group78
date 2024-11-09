import unittest
from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile
import os


class TestStorage(unittest.TestCase):
    """
    Unit tests for the LocalStorage class
    """

    def setUp(self):
        """
        Set up the test environment by creating a temporary directory
        for storage and initializing a LocalStorage instance.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """
        Test the initialization of the LocalStorage instance
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """
        Test the 'save' and 'load' functionality
        by saving a byte sequence to a specific
        key and verifying that it can be retrieved correctly.
        Also tests the behavior when
        trying to load a non-existent key, expecting a NotFoundError.
        """
        # key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)

        otherkey = f"test{os.sep}otherpath"
        with self.assertRaises(NotFoundError):
            self.storage.load(otherkey)

    def test_delete(self):
        """
        Test the 'delete' functionality
        """
        # key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        with self.assertRaises(NotFoundError):
            self.storage.load(key)

    def test_list(self):
        """
        Test the 'list' functionality
        """
        # key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [(f"test{os.sep}"
                        f"{random.randint(0, 100)}") for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)

        keys = self.storage.list("test")
        keys = [f"{os.sep}".join(key.split(f"{os.sep}")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))


if __name__ == "__main__":
    unittest.main()
