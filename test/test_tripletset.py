import unittest
import sys
import os
import numpy as np

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.embedding import TFTripletset

class TestTFTripletset(unittest.TestCase):
    def setUp(self):
        self.empty = TFTripletset()
        self.data = [[1,2],[3,4],[5,6],[7,8],[9,10]]
        self.labels = [1,2,2,3,3]
        self.tripletset = TFTripletset(data=self.data, labels=self.labels)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=[1,2,3], labels=[[1,1], [2,2], [3,3]])
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=[1,2,3], labels=None)
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=None, labels=[[1,1], [2,2], [3,3]])

    def test_split(self):
        # Test only data
        self.tripletset = TFTripletset(data=np.arange(100), labels=np.arange(100))
        self.tripletset.set_batch_size(5, 3)
        train, val, test = self.tripletset.split(1, 0, 0, False)
        self.assertTrue(isinstance(train, TFTripletset))
        train.batch_size_ = 5
        train.batch_positives_count_ = 3
        self.assertTrue(val is None)
        self.assertTrue(test is None)

        # Test only labels
        self.tripletset.set_batch_size(5, 3)
        train, val, test = self.tripletset.split(0, 0.5, 0.5, False)
        self.assertTrue(train is None)
        self.assertTrue(isinstance(val, TFTripletset))
        val.batch_size_ = 5
        val.batch_positives_count_ = 3
        self.assertTrue(isinstance(test, TFTripletset))
        test.batch_size_ = 5
        test.batch_positives_count_ = 3

        # Test different rates
        self.tripletset.set_batch_size(5, 3)
        train, val, test = self.tripletset.split(0.33, 0.33, 0.34, False)
        self.assertTrue(isinstance(train, TFTripletset))
        train.batch_size_ = 5
        train.batch_positives_count_ = 3
        self.assertTrue(isinstance(val, TFTripletset))
        val.batch_size_ = 5
        val.batch_positives_count_ = 3
        self.assertTrue(isinstance(test, TFTripletset))
        test.batch_size_ = 5
        test.batch_positives_count_ = 3

    def test_batch_size(self):
        with self.assertRaises(AssertionError):
            self.tripletset.set_batch_size(1, -1)
        with self.assertRaises(AssertionError):
            self.tripletset.set_batch_size(1, 2)
        self.tripletset.set_batch_size(3, 1)
        self.assertEqual(self.tripletset.batch_positives_count_, 1)
        self.assertEqual(self.tripletset.batch_negatives_count_, 2)

    def test_next_batch(self):
        self.tripletset.set_batch_size(3, 1)
        batch = self.tripletset.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels_ == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels_ == 1) == 2)
        self.assertTrue(np.asarray(batch.labels_).ndim == 1)
        self.assertTrue(len(batch.labels_) == len(batch.data_))
        batch = self.tripletset.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels_ == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels_ == 1) == 2)
        self.assertTrue(np.asarray(batch.labels_).ndim == 1)
        self.assertTrue(len(batch.labels_) == len(batch.data_))
        batch = self.tripletset.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels_ == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels_ == 1) == 2)
        self.assertTrue(np.asarray(batch.labels_).ndim == 1)
        self.assertTrue(len(batch.labels_) == len(batch.data_))

if __name__ == '__main__':
    unittest.main()
