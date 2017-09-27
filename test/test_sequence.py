import unittest
import sys
import os
import numpy as np

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import TFSequence

class TestTFSequence(unittest.TestCase):
    def setUp(self):
        self.empty = TFSequence()
        self.incorrect_data = [[1,2,3], [[4],[5]]]
        self.data = [[[11,11],[12,12],[13,13],[14,14]],
                     [[21,21],[22,22],[23,23]],
                     [[21,21],[22,22],[23,23]],
                     [[31,31],[32,32],[33,33],[34,34]],
                     [[31,31],[32,32],[33,33],[34,34],[35,35]]]
        self.labels = [1,2,2,3,3]
        self.sequences = TFSequence(data=self.data, labels=self.labels)
        self.short1 = TFSequence(data=[[1,2,3], [4,5]])
        self.short2 = TFSequence(data=[[[1],[2],[3]], [[4],[5]]])

    def test_initialize(self):
        with self.assertRaises(AssertionError):
            self.empty.initialize(data=self.incorrect_data, labels=None)
        self.assertTrue(np.all(np.asarray(self.sequences.data_shape) == np.asarray([None, 2])))

    def test_lengths(self):
        with self.assertRaises(AssertionError):
            self.empty.lengths()
        self.assertTrue(np.all(self.sequences.lengths() == np.asarray([4, 3, 3, 4, 5])))

    def test_concatenated_data(self):
        with self.assertRaises(AssertionError):
            self.empty.concatenated_data()
        self.assertTrue(np.all(self.short1.concatenated_data() == np.asarray([1, 2, 3, 4, 5])))
        self.assertTrue(np.all(self.short2.concatenated_data() == np.asarray([[1], [2], [3], [4], [5]])))

    def test_normalize_global(self):
        with self.assertRaises(AssertionError):
            self.empty.normalize(True)
        self.short1.normalize(True)
        self.assertTrue(np.sum(np.abs(self.short1.data[0] - np.asarray([-1.41421356, -0.70710678, 0]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short1.data[1] - np.asarray([0.70710678, 1.41421356]))) < 0.000001)
        self.assertEqual(len(self.short1.data), 2)

        self.short2.normalize(True)
        self.assertTrue(np.sum(np.abs(self.short2.data[0] - np.asarray([[-1.41421356], [-0.70710678], [0]]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short2.data[1] - np.asarray([[0.70710678], [1.41421356]]))) < 0.000001)
        self.assertEqual(len(self.short2.data), 2)

    def test_normalize_not_global(self):
        with self.assertRaises(AssertionError):
            self.empty.normalize(False)
        self.short1.normalize(False)
        self.assertTrue(np.sum(np.abs(self.short1.data[0] - np.asarray([-1.41421356, -0.70710678, 0]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short1.data[1] - np.asarray([0.70710678, 1.41421356]))) < 0.000001)
        self.assertEqual(len(self.short1.data), 2)

        self.short2.normalize(False)
        self.assertTrue(np.sum(np.abs(self.short2.data[0] - np.asarray([[-1.41421356], [-0.70710678], [0]]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short2.data[1] - np.asarray([[0.70710678], [1.41421356]]))) < 0.000001)
        self.assertEqual(len(self.short2.data), 2)

    def test_unnormalize_global(self):
        with self.assertRaises(AssertionError):
            self.empty.unnormalize()
        self.short1.normalize(True)
        self.short1.unnormalize()
        self.assertTrue(np.sum(np.abs(self.short1.data[0] - np.asarray([1,2,3]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short1.data[1] - np.asarray([4,5]))) < 0.000001)
        self.assertEqual(len(self.short1.data), 2)

        self.short2.normalize(True)
        self.short2.unnormalize()
        self.assertTrue(np.sum(np.abs(self.short2.data[0] - np.asarray([[1], [2], [3]]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short2.data[1] - np.asarray([[4], [5]]))) < 0.000001)
        self.assertEqual(len(self.short2.data), 2)

    def test_unnormalize_not_global(self):
        with self.assertRaises(AssertionError):
            self.empty.unnormalize(False)
        self.short1.normalize(False)
        self.short1.unnormalize()
        self.assertTrue(np.sum(np.abs(self.short1.data[0] - np.asarray([1,2,3]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short1.data[1] - np.asarray([4,5]))) < 0.000001)
        self.assertEqual(len(self.short1.data), 2)

        self.short2.normalize(False)
        self.short2.unnormalize()
        self.assertTrue(np.sum(np.abs(self.short2.data[0] - np.asarray([[1], [2], [3]]))) < 0.000001)
        self.assertTrue(np.sum(np.abs(self.short2.data[1] - np.asarray([[4], [5]]))) < 0.000001)
        self.assertEqual(len(self.short2.data), 2)

    def test_padding(self):
        sequences_with_padding, lengths = TFSequence.padding(self.data, 6)
        self.assertTrue(np.all(lengths == np.asarray([4, 3, 3, 4, 5])))
        sequences_with_padding, lengths = TFSequence.padding(self.data, 1)
        self.assertTrue(np.all(lengths == np.asarray([1, 1, 1, 1, 1])))
        sequences_with_padding, lengths = TFSequence.padding([[], [1,2,3]], 4)
        self.assertTrue(np.all(lengths == np.asarray([0, 3])))

if __name__ == '__main__':
    unittest.main()
