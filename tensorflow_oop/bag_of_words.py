"""
Bag of words.
"""

import numpy as np
from collections import Counter
import operator


class TFBagOfWords(object):

    """
    Bag of words model.
    """

    __slots__ = ['size',
                 'non_chars', 'lower_caise', 'digit_zero',
                 'min_count', 'max_count',
                 'words_counter', 'dictionary']

    def __init__(self,
                 texts,
                 non_chars='/.,!?()_-";:*=&|%<>@\'\t\n\r',
                 lower_caise=True,
                 digit_zero=True,
                 min_count=1,
                 max_count=np.inf):
        """Initialize Bag of words model.

        Arguments:
            texts -- list of texts for statistic
            non_chars -- symbols for excluding
            lower_caise -- mode with lower symbols
            digit_zero -- convert all digits to zero
            min_count -- filter minimum words count for adding to dictionary
            max_count -- filter maximum words count for adding to dictionary

        Return:
            words -- list

        """
        # Save properties
        self.non_chars = non_chars
        self.lower_caise = lower_caise
        self.digit_zero = digit_zero
        self.min_count = min_count
        self.max_count = max_count

        # Calculate statistic
        words = self.list_of_words(' '.join(texts))
        self.words_counter = Counter(words)

        # Calculate dictionary
        self.dictionary = {}
        for word in sorted(self.words_counter):
            if self.min_count <= self.words_counter[word] <= self.max_count:
                self.dictionary[word] = len(self.dictionary)
        self.size = len(self.dictionary)

    def list_of_words(self, text):
        """Get list of standart words from text.

        Arguments:
            text -- input value

        Return:
            words -- list

        """
        words = self._preprocessing(text).split(' ')
        if '' in words:
            words.remove('')
        return words

    def vectorize(self, text, binary):
        """Calculate vector by text.

        Arguments:
            text -- conversation text
            binary -- use rational or only [0, 1]

        Return:
            vector -- array

        """
        # Calculate statistic
        words = self.list_of_words(text)
        vector = np.zeros(self.size)
        for word in words:
            if word in self.dictionary:
                index = self.dictionary[word]
                if binary:
                    vector[index] = 1.
                else:
                    vector[index] += 1.

        # Validate data
        valid_count = np.sum(vector)
        assert valid_count > 0, \
            '''Valid words count should be greater then zero:
            valid_count = %s''' % valid_count

        # Normalize if necessary
        if not binary:
            vector /= valid_count
        return vector

    def __len__(self):
        return self.size

    def __str__(self):
        string = 'TFBagOfWords object:\n'
        for attr in self.__slots__:
            if attr != 'words_counter' and attr != 'dictionary':
                string += '%20s: %s\n' % (attr, getattr(self, attr))
        sorted_words_counter = sorted(self.words_counter.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
        string += '%20s:\n%s\n' % ('words_counter', sorted_words_counter)
        string += '%20s:\n%s\n' % ('dictionary', self.dictionary)
        return string[:-1]

    def _preprocessing(self, old_text):
        """Standartize text to one format.

        Arguments:
            old_text -- input value

        Return:
            new_text -- string

        """
        if self.lower_caise:
            new_text = old_text.lower()
        for non_char in self.non_chars:
            new_text = new_text.replace(non_char, ' ')
        if self.digit_zero:
            for i in range(1, 10):
                new_text = new_text.replace(str(i), '0')
        while new_text.find('  ') >= 0:
            new_text = new_text.replace('  ', ' ')
        new_text = new_text.strip()
        return new_text
