import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline

if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.loadTestsFromTestCase(TestDatabase)
    loader.loadTestsFromTestCase(TestStorage)
    loader.loadTestsFromTestCase(TestFeatures)
    loader.loadTestsFromTestCase(TestPipeline)
    unittest.main()
