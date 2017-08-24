import unittest
import sys
import os
import numpy as np

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import TFDataset

class TestTFDataset(unittest.TestCase):
    def setUp(self):
        self.empty = TFDataset()
        self.data = [[1,2],[3,4],[5,6]]
        self.labels = [1,2,3]
        self.dataset1 = TFDataset(data=self.data)
        self.dataset2 = TFDataset(labels=self.labels)
        self.dataset3 = TFDataset(data=self.data, labels=self.labels)

    def test_init(self):
        self.assertFalse(self.empty.init_)
        self.assertEqual(self.empty.size_, 0)
        self.assertEqual(self.empty.batch_size_, 1)
        self.assertEqual(self.empty.batch_num_, 0)

        self.assertTrue(self.dataset1.init_)
        self.assertEqual(self.dataset1.size_, 3)
        self.assertEqual(self.dataset1.batch_size_, 1)
        self.assertEqual(self.dataset1.batch_num_, 0)

        self.assertTrue(self.dataset2.init_)
        self.assertEqual(self.dataset2.size_, 3)
        self.assertEqual(self.dataset2.batch_size_, 1)
        self.assertEqual(self.dataset2.batch_num_, 0)

        self.assertTrue(self.dataset3.init_)
        self.assertEqual(self.dataset3.size_, 3)
        self.assertEqual(self.dataset3.batch_size_, 1)
        self.assertEqual(self.dataset3.batch_num_, 0)

    def test_data_initialize(self):
        with self.assertRaises(AssertionError):
            self.empty.initialize(None, None)
        with self.assertRaises(AssertionError):
            self.empty.initialize([1,2], [1,2,3])
        self.empty.initialize(self.data, None)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, [2])
        self.assertEqual(self.empty.data_ndim_, 1)
        self.assertEqual(self.empty.labels_shape_, None)
        self.assertEqual(self.empty.labels_ndim_, None)
        self.assertTrue(self.empty.init_)

    def test_labels_initialize(self):
        self.empty.initialize(None, self.labels)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, None)
        self.assertEqual(self.empty.data_ndim_, None)
        self.assertEqual(self.empty.labels_shape_, [1])
        self.assertEqual(self.empty.labels_ndim_, 1)
        self.assertTrue(self.empty.init_)

        self.dataset3.normalize()
        self.dataset3.initialize(None, self.labels)
        self.assertEqual(self.dataset3.normalized_, None)
        self.assertEqual(self.dataset3.normalization_mask_, None)
        self.assertEqual(self.dataset3.normalization_mean_, None)
        self.assertEqual(self.dataset3.normalization_std_, None)

    def test_data_labels_initialize(self):
        self.empty.initialize(self.data, self.labels)
        self.assertEqual(self.empty.size_, 3)
        self.assertEqual(self.empty.data_shape_, [2])
        self.assertEqual(self.empty.data_ndim_, 1)
        self.assertEqual(self.empty.labels_shape_, [1])
        self.assertEqual(self.empty.labels_ndim_, 1)
        self.assertTrue(self.empty.init_)

    def test_copy(self):
        self.empty.copy(self.dataset3)
        for attr in self.empty.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(self.empty, attr)) == np.asarray(getattr(self.dataset3, attr))))

    def test_shuffle(self):
        with self.assertRaises(AssertionError):
            self.empty.shuffle()

    def test_batch_size(self):
        with self.assertRaises(AssertionError):
            self.empty.set_batch_size(0)
        with self.assertRaises(AssertionError):
            self.empty.set_batch_size(1)
        with self.assertRaises(AssertionError):
            self.dataset1.set_batch_size(0)
        with self.assertRaises(AssertionError):
            self.dataset1.set_batch_size(5)
        self.dataset1.set_batch_size(2)
        self.assertEqual(self.dataset1.batch_size_, 2)

    def test_next_batch(self):
        with self.assertRaises(AssertionError):
            self.empty.next_batch(0)

        # Test only data
        batch = self.dataset1.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[1,2]])))
        self.assertEqual(batch.labels_, None)
        batch = self.dataset1.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[3,4]])))
        self.assertEqual(batch.labels_, None)
        batch = self.dataset1.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[5,6]])))
        self.assertEqual(batch.labels_, None)
        batch = self.dataset1.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[1,2]])))
        self.assertEqual(batch.labels_, None)

        # Test only labels
        batch = self.dataset2.next_batch()
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[1]])))
        self.assertEqual(batch.data_, None)
        batch = self.dataset2.next_batch()
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[2]])))
        self.assertEqual(batch.data_, None)
        batch = self.dataset2.next_batch()
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[3]])))
        self.assertEqual(batch.data_, None)
        batch = self.dataset2.next_batch()
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[1]])))
        self.assertEqual(batch.data_, None)

        # Test another batch size
        self.dataset3.set_batch_size(2)
        batch = self.dataset3.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[1,2],[3,4]])))
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[1],[2]])))
        batch = self.dataset3.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[5,6],[1,2]])))
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[3],[1]])))
        batch = self.dataset3.next_batch()
        self.assertTrue(np.all(np.asarray(batch.data_) == np.asarray([[3,4],[5,6]])))
        self.assertTrue(np.all(np.asarray(batch.labels_) == np.asarray([[2],[3]])))

    def test_split(self):
        with self.assertRaises(AssertionError):
            self.empty.split(1, 0, 0, False)
        with self.assertRaises(AssertionError):
            self.dataset1.split(-1, 0, 0, False)
        with self.assertRaises(AssertionError):
            self.dataset1.split(0, -1, 0, False)
        with self.assertRaises(AssertionError):
            self.dataset1.split(0, 0, -1, False)
        with self.assertRaises(AssertionError):
            self.dataset1.split(1, 0, 1, False)

        # Test only data
        train, val, test = self.dataset1.split(1, 0, 0, False)
        self.assertTrue(isinstance(train, TFDataset))
        self.assertEqual(len(train), 3)
        self.assertEqual(val, None)
        self.assertEqual(test, None)
        for attr in train.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(train, attr)) == np.asarray(getattr(self.dataset1, attr))))
        train, val, test = self.dataset1.split(3, 0, 0, False)
        self.assertTrue(isinstance(train, TFDataset))
        self.assertEqual(len(train), 3)
        self.assertEqual(val, None)
        self.assertEqual(test, None)
        for attr in train.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(train, attr)) == np.asarray(getattr(self.dataset1, attr))))

        # Test only labels
        train, val, test = self.dataset2.split(0, 1, 0, False)
        self.assertEqual(train, None)
        self.assertTrue(isinstance(val, TFDataset))
        self.assertEqual(len(val), 3)
        self.assertEqual(test, None)
        for attr in val.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(val, attr)) == np.asarray(getattr(self.dataset2, attr))))
        train, val, test = self.dataset2.split(0, 3, 0, False)
        self.assertEqual(train, None)
        self.assertTrue(isinstance(val, TFDataset))
        self.assertEqual(len(val), 3)
        self.assertEqual(test, None)
        for attr in val.__slots__:
            self.assertTrue(np.all(np.asarray(getattr(val, attr)) == np.asarray(getattr(self.dataset2, attr))))

        # Test different rates
        train, val, test = self.dataset3.split(0.33, 0.33, 0.34, False)
        self.assertTrue(isinstance(train, TFDataset))
        self.assertEqual(len(train), 1)
        self.assertTrue(isinstance(val, TFDataset))
        self.assertEqual(len(val), 1)
        self.assertTrue(isinstance(test, TFDataset))
        self.assertEqual(len(test), 1)
        train, val, test = self.dataset3.split(1, 1, 1, False)
        self.assertTrue(isinstance(train, TFDataset))
        self.assertEqual(len(train), 1)
        self.assertTrue(isinstance(val, TFDataset))
        self.assertEqual(len(val), 1)
        self.assertTrue(isinstance(test, TFDataset))
        self.assertEqual(len(test), 1)

        # Test shuffling
        train, val, test = self.dataset3.split(1, 1, 1, True)
        self.assertTrue(isinstance(train, TFDataset))
        self.assertEqual(len(train), 1)
        self.assertTrue(isinstance(val, TFDataset))
        self.assertEqual(len(val), 1)
        self.assertTrue(isinstance(test, TFDataset))
        self.assertEqual(len(test), 1)

    def test_generate_sequences(self):
        with self.assertRaises(AssertionError):
            self.empty.generate_sequences(1, 1)
        with self.assertRaises(AssertionError):
            self.dataset2.generate_sequences(1, 1)
        with self.assertRaises(AssertionError):
            self.dataset3.generate_sequences(0, 1)
        with self.assertRaises(AssertionError):
            self.dataset3.generate_sequences(1, 0)
        with self.assertRaises(AssertionError):
            self.dataset3.generate_sequences(1, 1, 0, 1)
        with self.assertRaises(AssertionError):
            self.dataset3.generate_sequences(1, 1, 1)

        # Test only data
        seq = self.dataset3.generate_sequences(1, 1)
        self.assertTrue(isinstance(seq, TFDataset))
        self.assertEqual(len(seq), 3)
        self.assertTrue(np.all(np.asarray(seq.data_shape_) == np.asarray([1, 2])))
        self.assertEqual(seq.data_ndim_, 2)
        self.assertEqual(seq.labels_, None)

        # Test data with labels
        seq = self.dataset1.generate_sequences(2, 1, 1, 0)
        self.assertTrue(isinstance(seq, TFDataset))
        self.assertEqual(len(seq), 1)
        self.assertTrue(np.all(np.asarray(seq.data_shape_) == np.asarray([2, 2])))
        self.assertEqual(seq.data_ndim_, 2)
        self.assertTrue(np.all(np.asarray(seq.labels_shape_) == np.asarray([1, 2])))
        self.assertEqual(seq.labels_ndim_, 2)

    def test_normalize(self):
        # Test without mask
        with self.assertRaises(AssertionError):
            self.empty.normalize()
        with self.assertRaises(AssertionError):
            self.dataset2.normalize()
        self.dataset1.normalize()
        self.assertTrue(self.dataset1.normalized_)
        self.assertEqual(self.dataset1.normalization_mask_, None)
        self.assertEqual(self.dataset1.normalization_mean_, np.mean(self.data))
        self.assertEqual(self.dataset1.normalization_std_, np.std(self.data))
        
        # Test with mask
        dataset = TFDataset(data=[[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]])
        with self.assertRaises(AssertionError):
            dataset.normalize(mask=[True])
        with self.assertRaises(AssertionError):
            dataset.normalize(mask=[False, False])
        with self.assertRaises(AssertionError):
            dataset.normalize(mask=[True, False])
        dataset.normalize(mask=[False, True])
        self.assertTrue(dataset.normalized_)
        self.assertTrue(np.all(np.asarray(dataset.normalization_mask_) == np.asarray([False, True])))
        self.assertTrue(np.all(dataset.normalization_mean_ == np.asarray([np.mean([1,3,5,7,9,11]), np.mean([2,4,6,8,10,12])])))
        self.assertTrue(np.all(dataset.normalization_std_  == np.asarray([np.std([1,3,5,7,9,11]),  np.std([2,4,6,8,10,12])])))

        self.dataset3.normalize(mask=[True])
        self.assertTrue(self.dataset3.normalized_)
        self.assertTrue(np.all(np.asarray(self.dataset3.normalization_mask_) == np.asarray([True])))
        self.assertTrue(np.all(self.dataset3.normalization_mean_ == np.asarray([np.mean([1,3,5]), np.mean([2,4,6])])))
        self.assertTrue(np.all(self.dataset3.normalization_std_  == np.asarray([np.std([1,3,5]),  np.std([2,4,6])])))
        
    def test_unnormalize(self):
        with self.assertRaises(AssertionError):
            self.empty.unnormalize()
        self.dataset1.normalize()
        self.dataset1.unnormalize()
        self.assertFalse(self.dataset1.normalized_)
        self.assertTrue(np.all(self.dataset1.data_ == self.data))

        self.dataset3.normalize()
        self.dataset3.unnormalize()
        self.assertFalse(self.dataset3.normalized_)
        self.assertTrue(np.all(self.dataset3.data_ == self.data))

    def test_one_hot(self):
        with self.assertRaises(AssertionError):
            self.empty.one_hot(10)
        with self.assertRaises(AssertionError):
            self.dataset1.one_hot(10)
        with self.assertRaises(AssertionError):
            self.dataset2.one_hot(1)
        dataset = TFDataset(data=[1,2,3], labels=[0.1, 0.2, 0.3])
        with self.assertRaises(AssertionError):
            dataset.one_hot(10)
        dataset = TFDataset(data=[1,2,3], labels=[[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        with self.assertRaises(AssertionError):
            dataset.one_hot(10)
        self.dataset2.one_hot(4)
        self.assertTrue(np.all(np.asarray(self.dataset2.labels_) == np.asarray([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])))
        self.dataset3.one_hot(5)
        self.assertTrue(np.all(np.asarray(self.dataset3.labels_) == np.asarray([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])))

if __name__ == '__main__':
    unittest.main()
