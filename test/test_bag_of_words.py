import unittest
import sys
import os
import numpy as np

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.bag_of_words import TFBagOfWords

class TestTFBagOfWords(unittest.TestCase):
    def setUp(self):
        self.texts = ['Hello World!', '1 12 123 456', ' ;  @  : ']
        self.empty = TFBagOfWords([])
        self.bow = TFBagOfWords(self.texts)
        self.bow_all = TFBagOfWords(self.texts, '')

    def test_init(self):
        self.assertEqual(self.empty.size, 0)
        self.assertEqual(self.bow.size, 5)
        self.assertEqual(self.bow_all.size, 8)

    def test_list_of_words(self):
        list_of_words = self.bow.list_of_words(self.texts[0])
        self.assertEqual(list_of_words[0], 'hello')
        self.assertEqual(list_of_words[1], 'world')
        self.assertEqual(len(list_of_words), 2)

        list_of_words = self.bow.list_of_words(self.texts[1])
        self.assertEqual(list_of_words[0], '0')
        self.assertEqual(list_of_words[1], '00')
        self.assertEqual(list_of_words[2], '000')
        self.assertEqual(list_of_words[3], '000')
        self.assertEqual(len(list_of_words), 4)

        list_of_words = self.bow_all.list_of_words(self.texts[2])
        self.assertEqual(list_of_words[0], ';')
        self.assertEqual(list_of_words[1], '@')
        self.assertEqual(list_of_words[2], ':')
        self.assertEqual(len(list_of_words), 3)

    def test_vectorize(self):
        with self.assertRaises(AssertionError):
            self.empty.vectorize(self.texts[0], True)
        with self.assertRaises(AssertionError):
            self.empty.vectorize(self.texts[0], False)

        vector = self.bow.vectorize(self.texts[0], True)
        self.assertTrue(np.all(vector == [0, 0, 0, 1, 1]))
        vector = self.bow.vectorize(self.texts[0], False)
        self.assertTrue(np.all(vector == [0, 0, 0, 0.5, 0.5]))

        vector = self.bow.vectorize(self.texts[1], True)
        self.assertTrue(np.all(vector == [1, 1, 1, 0, 0]))
        vector = self.bow.vectorize(self.texts[1], False)
        self.assertTrue(np.all(vector == [0.25, 0.25, 0.5, 0, 0]))

        with self.assertRaises(AssertionError):
            self.bow.vectorize(self.texts[2], True)
        with self.assertRaises(AssertionError):
            self.bow.vectorize(self.texts[2], False)

        vector = self.bow_all.vectorize(self.texts[2], True)
        self.assertTrue(np.all(vector == [0, 0, 0, 1, 1, 1, 0, 0]))
        vector = self.bow_all.vectorize(self.texts[2], False)
        self.assertTrue(np.all(vector == [0, 0, 0, 1./3., 1./3., 1./3., 0, 0]))

if __name__ == '__main__':
    unittest.main()
