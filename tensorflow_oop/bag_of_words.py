"""
Bag of words.
"""

import numpy as np
from collections import Counter
import operator


class TFBagOfWords(object):

    """
    Bag of words model.

    Attributes:
        size               Dictionary length.
        non_chars          String with chars for excluding.
        lower_case         Indicator of case insensitive mode.
        digit_as_zero      Indicator of converting all digits to zero.
        min_count          Filter minimum words count for adding to dictionary.
        max_count          Filter maximum words count for adding to dictionary.
        words_counter      Counter object of all words in texts.
        dictionary         Dict object of correct words in texts.
    """

    __slots__ = ['size',
                 'non_chars', 'lower_case', 'digit_as_zero',
                 'min_count', 'max_count',
                 'words_counter', 'dictionary']

    def __init__(self,
                 texts,
                 non_chars=None,
                 lower_case=True,
                 digit_as_zero=True,
                 min_count=1,
                 max_count=np.inf):
        """Constructor.

        Arguments:
            texts              List of texts for building dictionary.
            non_chars          String with chars for excluding.
            lower_case         Indicator of case insensitive mode.
            digit_as_zero      Indicator of converting all digits to zero.
            min_count          Filter minimum words count for adding to dictionary.
            max_count          Filter maximum words count for adding to dictionary.

        """

        # Save properties
        if non_chars is not None:
            self.non_chars = non_chars
        else:
            self.non_chars = '/.,!?()_-";:*=&|%<>@\'\t\n\r'
        self.lower_case = lower_case
        self.digit_as_zero = digit_as_zero
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
            text        Text in string format.

        Return:
            words       List of extracted words.

        """
        words = self.preprocessing(text).split(' ')
        if '' in words:
            words.remove('')
        return words

    def vectorize(self, text, binary):
        """Calculate vector by text.

        Arguments:
            text        Conversation text.
            binary      Indicator of using only {0, 1} instead rational from [0, 1].

        Return:
            vector      Numeric representation vector.

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

    def preprocessing(self, old_text):
        """Standartize text to one format.

        Arguments:
            old_text    Text to preprocessing in string format.

        Return:
            new_text    Processed text.

        """
        if self.lower_case:
            new_text = old_text.lower()
        for non_char in self.non_chars:
            new_text = new_text.replace(non_char, ' ')
        if self.digit_as_zero:
            for i in range(1, 10):
                new_text = new_text.replace(str(i), '0')
        while new_text.find('  ') >= 0:
            new_text = new_text.replace('  ', ' ')
        new_text = new_text.strip()
        return new_text

    def __len__(self):
        """Unique words count in dictionary."""
        return self.size

    def __str__(self):
        """String formatting."""
        string = 'TFBagOfWords object:\n'
        for attr in self.__slots__:
            if attr != 'words_counter' and attr != 'dictionary':
                string += '%20s: %s\n' % (attr, getattr(self, attr))
        sorted_words_counter = sorted(self.words_counter.items(),
                                      key=operator.itemgetter(1),
                                      reverse=True)
        string += '%20s:\n%s\n...\n%s\n' % ('words_counter',
            '\n'.join([str(elem) for elem in sorted_words_counter[:10]]),
            '\n'.join([str(elem) for elem in sorted_words_counter[-10:]]))
        sorted_dictionary = sorted(self.dictionary.items(),
                                   key=operator.itemgetter(0),
                                   reverse=False)
        string += '%20s:\n%s\n...\n%s\n' % ('dictionary',
            '\n'.join([str(elem) for elem in sorted_dictionary[:10]]),
            '\n'.join([str(elem) for elem in sorted_dictionary[-10:]]))
        return string[:-1]
