import unittest
import sys
import os
import numpy as np

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.embedding import TFTripletSequence

class TestTFTripletSequence(unittest.TestCase):
    def setUp(self):
        self.empty = TFTripletSequence()
        self.data = [[[11,11],[12,12],[13,13],[14,14]],
                     [[21,21],[22,22],[23,23]],
                     [[21,21],[22,22],[23,23]],
                     [[31,31],[32,32],[33,33],[34,34]],
                     [[31,31],[32,32],[33,33],[34,34],[35,35]]]
        self.labels = [1,2,2,3,3]
        self.tripletseq = TFTripletSequence(data=self.data, labels=self.labels)

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=[1,2,3], labels=[[1,1], [2,2], [3,3]])
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=[1,2,3], labels=None)
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=None, labels=[[1,1], [2,2], [3,3]])

    def test_split(self):
        # Test only data
        self.tripletseq = TFTripletSequence(data=np.arange(100),
                                            labels=np.arange(100))
        self.tripletseq.set_batch_size(5, 3, 2)
        train, val, test = self.tripletseq.split(1, 0, 0, False)
        self.assertTrue(isinstance(train, TFTripletSequence))
        self.assertEqual(train.batch_size, 5)
        self.assertEqual(train.batch_positives_count, 3)
        self.assertEqual(train.batch_negatives_count, 2)
        self.assertEqual(train.max_sequence_length, 2)
        self.assertTrue(val is None)
        self.assertTrue(test is None)

        # Test only labels
        self.tripletseq.set_batch_size(5, 3, 2)
        train, val, test = self.tripletseq.split(0, 0.5, 0.5, False)
        self.assertTrue(train is None)
        self.assertTrue(isinstance(val, TFTripletSequence))
        self.assertEqual(val.batch_size, 5)
        self.assertEqual(val.batch_positives_count, 3)
        self.assertEqual(val.batch_negatives_count, 2)
        self.assertEqual(val.max_sequence_length, 2)
        self.assertTrue(isinstance(test, TFTripletSequence))
        self.assertEqual(test.batch_size, 5)
        self.assertEqual(test.batch_positives_count, 3)
        self.assertEqual(test.batch_negatives_count, 2)
        self.assertEqual(test.max_sequence_length, 2)

        # Test different rates
        self.tripletseq.set_batch_size(5, 3, 2)
        train, val, test = self.tripletseq.split(0.33, 0.33, 0.34, False)
        self.assertTrue(isinstance(train, TFTripletSequence))
        self.assertEqual(train.batch_size, 5)
        self.assertEqual(train.batch_positives_count, 3)
        self.assertEqual(train.batch_negatives_count, 2)
        self.assertEqual(train.max_sequence_length, 2)
        self.assertTrue(isinstance(val, TFTripletSequence))
        self.assertEqual(val.batch_size, 5)
        self.assertEqual(val.batch_positives_count, 3)
        self.assertEqual(val.batch_negatives_count, 2)
        self.assertEqual(val.max_sequence_length, 2)
        self.assertTrue(isinstance(test, TFTripletSequence))
        self.assertEqual(test.batch_size, 5)
        self.assertEqual(test.batch_positives_count, 3)
        self.assertEqual(test.batch_negatives_count, 2)
        self.assertEqual(test.max_sequence_length, 2)

    def test_batch_size(self):
        with self.assertRaises(AssertionError):
            self.tripletseq.set_batch_size(1, -1, 1)
        with self.assertRaises(AssertionError):
            self.tripletseq.set_batch_size(1, 2, 1)
        with self.assertRaises(AssertionError):
            self.tripletseq.set_batch_size(3, 1, -1)
        self.tripletseq.set_batch_size(3, 1, 2)
        self.assertEqual(self.tripletseq.batch_size, 3)
        self.assertEqual(self.tripletseq.batch_positives_count, 1)
        self.assertEqual(self.tripletseq.batch_negatives_count, 2)
        self.assertEqual(self.tripletseq.max_sequence_length, 2)

    def test_next_batch(self):
        self.tripletseq.set_batch_size(3, 1, 2)
        batch = self.tripletseq.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels == 1) == 2)
        self.assertTrue(np.asarray(batch.labels).ndim == 1)
        self.assertTrue(len(batch.labels) == len(batch.data))
        batch = self.tripletseq.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels == 1) == 2)
        self.assertTrue(np.asarray(batch.labels).ndim == 1)
        self.assertTrue(len(batch.labels) == len(batch.data))
        batch = self.tripletseq.next_batch()
        self.assertTrue(np.count_nonzero(batch.labels == 0) == 1)
        self.assertTrue(np.count_nonzero(batch.labels == 1) == 2)
        self.assertTrue(np.asarray(batch.labels).ndim == 1)
        self.assertTrue(len(batch.labels) == len(batch.data))

if __name__ == '__main__':
    unittest.main()
