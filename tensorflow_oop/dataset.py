"""
Dataset and batch base classes.
"""

import numpy as np
import pickle
import warnings


def check_initialization(function):
    """Decorator for check initialization."""
    def wrapper(self, *args, **kwargs):
        assert self.init, \
            'Object should be initialized: self.init = %s' % self.init
        return function(self, *args, **kwargs)
    return wrapper


def update_last_batch(function):
    """Decorator for updating last batch."""
    def wrapper(self, *args, **kwargs):
        batch = function(self, *args, **kwargs)
        self.last_batch = batch
        return batch
    return wrapper


class TFBatch:

    """
    Batch container.
    """

    def __init__(self, **kwargs):
        """Constructor.

        Arguments:
            kwargs      Dict of arguments for store as batch atributes.

        """
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        """String formatting."""
        string = 'TFBatch object:\n'
        for attr in self.__dict__:
            string += '%20s:\n%s\n' % (attr, getattr(self, attr))
        return string[:-1]


class TFDataset(object):

    """
    Dataset of features.
    Data shape description:   [size, data_shape]
    Labels shape description: [size, labels_shape]

    Attributes:
        init               Indicator of initializing dataset.
        size               Size of dataset.
        data               Array like data values.
        data_shape         Shape of data without first dimension.
        data_ndim          Data dimensions count without first dimension.
        labels             Array like labels values.
        labels_shape       Shape of labels without first dimension.
        labels_ndim        Labels dimensions count without first dimension.
        batch_size         Elements count per batch.
        batch_num          Numeric batch counter.
        last_batch         Last generated batch (not full batch).
        normalized         Indicator of normalizing dataset.
        norm_global        Indicator of global normalization.
        norm_mean          Mean value of data before normalizing.
        norm_std           Standart deviation value of data before normalizing.

    """

    __slots__ = ['init', 'size',
                 'data', 'data_shape', 'data_ndim',
                 'labels', 'labels_shape', 'labels_ndim',
                 'batch_size', 'batch_num', 'last_batch',
                 'normalized', 'norm_global', 'norm_mean', 'norm_std']

    def __init__(self, data=None, labels=None):
        """Constructor.

        Arguments:
            data        Array like values for store as data attribute.
            labels      Array like values for store as labels attribute.

        """
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.init = False
        self.size = 0
        self.batch_size = 1
        self.batch_num = 0
        if data is not None or labels is not None:
            self.initialize(data=data, labels=labels)

    def copy(self, other):
        """Copy other dataset.

        Arguments:
            other       Other dataset.

        """
        for attr in self.__slots__:
            setattr(self, attr, getattr(other, attr))

    def initialize(self, data, labels):
        """Set data and labels.

        Arguments:
            data        Array like values for store as data attribute.
            labels      Array like values for store as labels attribute.

        """

        assert data is not None or labels is not None, \
            '''Data or labels should be passed:
            data = %s, labels = %s''' % (data, labels)

        if data is not None and labels is not None:
            assert len(data) == len(labels), \
                '''Data and labels should be the same length:
                len(data) = %s, len(labels) = %s''' % (len(data), len(labels))

        # Processing data
        if data is not None:
            self._set_data(data)
        else:
            self._reset_data()

        # Processing labels
        if labels is not None:
            self._set_labels(labels)
        else:
            self._reset_labels()

        if self.batch_size > self.size:
            self.batch_size = self.size
            warnings.warn('''Batch size automatically decreased to dataset size:
                             self.batch_size = %s''' % self.batch_size)
        self.init = True

    def _set_data(self, data):
        """Update data value.

        Arguments:
            data        Array like values for store as data attribute.

        """
        self.size = len(data)
        data = np.asarray(data)
        self.data = data
        self.data_shape = list(self.data.shape[1:])
        self.data_ndim = len(self.data_shape)

    def _reset_data(self):
        """Reset data to default value."""
        self.data = None
        self.data_shape = None
        self.data_ndim = None
        self.normalized = None
        self.norm_global = None
        self.norm_mean = None
        self.norm_std = None

    def _set_labels(self, labels):
        """Update labels value.

        Arguments:
            labels      Array like values for store as labels attribute.

        """
        self.size = len(labels)
        labels = np.asarray(labels)
        self.labels = labels
        self.labels_shape = list(self.labels.shape[1:])
        self.labels_ndim = len(self.labels_shape)

    def _reset_labels(self):
        """Reset labels to default value."""
        self.labels = None
        self.labels_shape = None
        self.labels_ndim = None

    @check_initialization
    def shuffle(self):
        """Random shuffling of dataset."""
        indexes = np.arange(self.size)
        np.random.shuffle(indexes)
        self.data = self.data[indexes]
        self.labels = self.labels[indexes]

    @check_initialization
    def set_batch_size(self, batch_size):
        """Set batch size.

        Arguments:
            batch_size  Elements count per batch.

        """

        assert batch_size > 0, \
            '''Batch size should be greater then zero:
            batch_size = %s''' % batch_size

        assert batch_size <=  self.size, \
            '''Batch size should not be greater then dataset size:
            batch_size = %s, self.size = %s''' % (batch_size, self.size)
        self.batch_size = int(batch_size)

    @check_initialization
    @update_last_batch
    def next_batch(self):
        """Get next batch."""

        # Get batch indexes
        first = (self.batch_num * self.batch_size) % self.size
        last = first + self.batch_size
        if (last <= self.size):
            batch_indexes = np.arange(first, last)
        else:
            batch_indexes = np.append(np.arange(first, self.size), np.arange(last - self.size))

        # Get batch data
        batch_data = None
        if self.data is not None:
            batch_data = self.data[batch_indexes]

        # Get batch labels
        batch_labels = None
        if self.labels is not None:
            batch_labels = self.labels[batch_indexes]
        self.batch_num += 1
        return TFBatch(data=batch_data, labels=batch_labels, indexes=batch_indexes)

    @check_initialization
    def full_batch(self):
        """Get full batch with size as dataset."""
        return TFBatch(data=self.data, labels=self.labels)

    @check_initialization
    def iterbatches(self, count=None):
        """Get iterator by batches.

        Arguments:
            count       Count of iterations, if not passed,
                        auto calculated batches count for one epoch.

        Return:
            batch       Iterator by batches.

        """
        if count is None:
            count = self.size // self.batch_size
            if self.size % self.batch_size != 0:
                count += 1
        for i in range(count):
            yield self.next_batch()

    @check_initialization
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set.

        Arguments:
            train_size  Count or rate of training size.
            val_size    Count or rate of validation size.
            test_size   Count or rate of testing size.
            shuffle     Indicator of randomizing indexes before splitting.

        Return:
            train_set   Dataset for training.
            val_set     Dataset for validation.
            test_set    Dataset for testing.

        """

        assert train_size >= 0, \
            '''Training size should not be less then zero:
            train_size = %s''' % train_size

        assert val_size >= 0, \
            '''Validation size should not be less then zero:
            val_size = %s''' % val_size

        assert test_size >= 0, \
            '''Testing size should not be less then zero:
            test_size = %s''' % test_size

        total_size = train_size + val_size + test_size
        assert total_size == self.size or total_size == 1, \
            '''Total size should be equal to TFDataset size or one:
            total_size = %s, self.size = %s''' % (total_size, self.size)

        # Check if arguments send as rate
        if total_size == 1:
            if train_size != 0:
                train_size = int(round(float(train_size) * self.size))
            else:
                train_size = 0
            if test_size != 0:
                if val_size != 0:
                    test_size = int(round(float(test_size) * self.size))
                else:
                    test_size = self.size - train_size
            else:
                test_size = 0
            if val_size != 0:
                val_size = self.size - train_size - test_size
            else:
                val_size = 0

        # Get indexes for don't dropping current data
        indexes = np.arange(self.size)

        # Shuffling if necessary
        if shuffle:
            np.random.shuffle(indexes)

        # Generate training set
        if train_size > 0:
            train_set = self.__class__()
            train_set.copy(self)
            if self.data is not None:
                data = self.data[indexes[:train_size]]
            else:
                data = None
            if self.labels is not None:
                labels = self.labels[indexes[:train_size]]
            else:
                labels = None
            train_set.initialize(data, labels)
        else:
            train_set = None

        # Generate validation set
        if val_size > 0:
            val_set = self.__class__()
            val_set.copy(self)
            if self.data is not None:
                data = self.data[indexes[train_size:train_size + val_size]]
            else:
                data = None
            if self.labels is not None:
                labels = self.labels[indexes[train_size:train_size + val_size]]
            else:
                labels = None
            val_set.initialize(data, labels)
        else:
            val_set = None

        # Generate testing set
        if test_size > 0:
            test_set = self.__class__()
            test_set.copy(self)
            if self.data is not None:
                data = self.data[indexes[-test_size:]]
            else:
                data = None
            if self.labels is not None:
                labels = self.labels[indexes[-test_size:]]
            else:
                labels = None
            test_set.initialize(data, labels)
        else:
            test_set = None

        return train_set, val_set, test_set

    @staticmethod
    def load(filename):
        """Load dataset from file.

        Arguments:
            filename    Path to loading dump.

        Return:
            obj         Loaded dataset.

        """
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, TFDataset), \
            '''Loaded object should be TFDataset object:
            type(obj) = %s''' % type(obj)
        return obj

    @check_initialization
    def save(self, filename):
        """Save dataset to file.

        Arguments:
            filename    Path to saving dump.

        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=-1)

    @check_initialization
    def generate_sequences(self,
                           sequence_length,
                           sequence_step,
                           label_length=None,
                           label_offset=None):
        """Generate sequences by slicing of dataset.

        Arguments:
            sequence_length    Length of output sequence data.
            sequence_step      Step before neighbor sequences data.
            label_length       Length of output sequence label.
            label_offset       Offset of sequence label begin after sequence data end.

        Return:
            sequences          Generated sequence data.
            labels             Generated sequence labels.

        """

        assert self.data is not None, \
            '''Data field should be initialized: self.data = %s''' % self.data

        assert sequence_length > 0, \
            '''Sequence length should be greater than zero:
            sequence_length = %s''' % sequence_length

        assert sequence_step > 0, \
            '''Sequence step should be greater than zero:
            sequence_step = %s''' % sequence_step

        if label_length is not None:
            assert label_length > 0 and label_offset is not None, \
                '''Label length should be greater than zero and
                label offset passed: label_length = %s,
                label_offset = %s''' % (label_length, label_offset)

        if label_length is not None:
            sequences = []
            labels = []
            last = self.size - sequence_length - label_length - label_offset + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data[i : last_sequence_index]
                sequences.append(current_sequence)
                first_ind = last_sequence_index + label_offset
                current_label = self.data[first_ind : first_ind + label_length]
                labels.append(current_label)
            return np.asarray(sequences), np.asarray(labels)
        else:
            sequences = []
            last = self.size - sequence_length + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data[i : last_sequence_index]
                sequences.append(current_sequence)
            return np.asarray(sequences), None

    @check_initialization
    def normalize(self, norm_global):
        """Normalize data to zero mean and one std.

        Arguments:
            norm_global        Indicator of global normalization.

        """
        assert self.data is not None, \
            '''Data field should be initialized: self.data = %s''' % self.data

        if self.normalized:
            return

        result_data = self._normalize_implementation(self.data,
                                                     self.data_shape,
                                                     norm_global)

        # Update dataset with normalized value
        self.data = result_data

    def _normalize_implementation(self, data, data_shape, norm_global):
        if not norm_global:
            # Reshape to array of feature vectors
            if len(data_shape) > 0:
                reshaped_data = np.reshape(data, [-1] + [np.sum(data_shape)])
            else:
                reshaped_data = np.reshape(data, [-1])
            reshaped_mean = np.mean(reshaped_data, axis=0)
            reshaped_std = np.std(reshaped_data, axis=0)
            
            # Add safe std for excluding division to zero
            safe_std = np.array(reshaped_std)
            safe_std[safe_std == 0.] = 1.

            normalized_data = (reshaped_data - reshaped_mean) / safe_std

            # Correct data shape and save normalization properties
            self.norm_mean = np.reshape(reshaped_mean, data_shape)
            self.norm_std = np.reshape(reshaped_std, data_shape)
            result_data = np.reshape(normalized_data, [-1] + data_shape)
        else:
            # Save normalization properties
            self.norm_mean = np.mean(data)
            self.norm_std = np.std(data)
            
            # Add safe std for excluding division to zero
            if self.norm_std == 0.:
                safe_std = 1.
            else:
                safe_std = self.norm_std

            # Update dataset with normalized value
            result_data = (data - self.norm_mean) / safe_std

        self.norm_global = norm_global
        self.normalized = True
        return result_data

    @check_initialization
    def unnormalize(self):
        """Unnormalize dataset to original from zero mean and one std."""

        assert self.data is not None, \
            'Data field should be initialized: self.data = %s' % self.data

        if not self.normalized:
            return

        result_data = self._unnormalize_implementation(self.data, self.data_shape)

        # Update dataset with unnormalized value
        self.data = result_data

    def _unnormalize_implementation(self, data, data_shape):
        if not self.norm_global:
            # Reshape normalization properties to vector
            if len(data_shape) > 0:
                reshaped_data = np.reshape(data, [-1] + [np.sum(data_shape)])
            else:
                reshaped_data = np.reshape(data, [-1])
            reshaped_mean = np.reshape(self.norm_mean, [-1])
            reshaped_std = np.reshape(self.norm_std, [-1])

            # Add safe std for excluding division to zero
            safe_std = np.array(reshaped_std)
            safe_std[safe_std == 0.] = 1.

            # Calculate unnormalized data
            unnormalized_data = reshaped_data*safe_std +  reshaped_mean
            result_data = np.reshape(unnormalized_data, [-1] + data_shape)
        else:
            # Add safe std for excluding division to zero
            if self.norm_std == 0.:
                safe_std = 1.
            else:
                safe_std = self.norm_std

            # Calculate unnormalized data
            result_data =  data*safe_std + self.norm_mean

        self.normalized = False
        return result_data

    @check_initialization
    def one_hot(self, encoding_size, dtype=np.float):
        """One hot encoding of labels.

        Arguments:
            encoding_size      Number of unique values for coding.
            dtype              Type of generated one hot labels.

        """

        assert self.labels is not None, \
            '''Labels should be initialized: self.labels = %s''' % self.labels

        assert np.issubdtype(self.labels.dtype, np.integer), \
            '''Labels type should be integer:
            self.labels.dtype = %s''' % self.labels.dtype

        assert np.min(self.labels) >= 0, \
            '''Minimal label should not be less than zero:
            np.min(self.labels) = %s''' % np.min(self.labels)

        assert encoding_size > np.max(self.labels), \
            '''Encoding size should be greater than maximal label:
            encoding_size = %s, np.max(self.labels) = %s''' \
            % (encoding_size, np.max(self.labels))

        flattened_labels = self.labels.flatten()
        assert len(flattened_labels) == self.size, \
            '''Flattened labels length should be equal to size of elements:
            len(flattened_labels) = %s, self.size = %s''' \
            % (len(flattened_labels), self.size)

        one_hot_labels = np.eye(encoding_size, dtype=dtype)[flattened_labels]
        self.initialize(data=self.data, labels=one_hot_labels)

    def str_shape(self):
        """Shape formatting as string."""
        return '%s : %s -> %s' % (self.size, self.data_shape, self.labels_shape)

    def __len__(self):
        """Size of dataset."""
        return self.size

    def __str__(self):
        """String formatting."""
        string = 'TFDataset object:\n'
        for attr in self.__slots__:
            if attr != 'data' and attr != 'labels':
                string += '%20s: %s\n' % (attr, getattr(self, attr))
        if 'data' in self.__slots__:
            string += '%s:\n%s\n' % ('data', getattr(self, 'data'))
        if 'labels' in self.__slots__:
            string += '%s:\n%s\n' % ('labels', getattr(self, 'labels'))
        return string[:-1]

class TFSequence(TFDataset):

    """
    Dataset of different length sequences of features.
    Data shape description:   [size, sequence_length, data_shape]
    Labels shape description: [size, labels_shape]

    Attributes:
        ...         Parrent class atributes.

    """

    def _set_data(self, data):
        """Update data value.

        Arguments:
            data        Array like values for store as data attribute.

        """
        unique_shapes = set([np.asarray(sequence).shape[1:] for sequence in data])
        assert len(unique_shapes) == 1, \
            '''All feature shapes in sequences should be equal:
            unique_shapes = %s''' % unique_shapes
        self.size = len(data)
        self.data = np.array([np.asarray(elem) for elem in data])
        self.data_shape = [None] + list(list(unique_shapes)[0])
        self.data_ndim = len(self.data_shape)

    @check_initialization
    def lengths(self):
        """Sequences lengths."""
        return np.array([len(sequence) for sequence in self.data])

    @check_initialization
    def concatenated_data(self):
        """Concatenate sequences to one array."""
        return np.concatenate(self.data)

    @check_initialization
    def normalize(self, norm_global):
        """Normalize data to zero mean and one std.

        Arguments:
            norm_global        Indicator of global normalization.

        """

        assert self.data is not None, \
            '''Data field should be initialized: self.data = %s''' % self.data

        if self.normalized:
            return

        result_data = self._normalize_implementation(self.concatenated_data(),
                                                     self.data_shape[1:],
                                                     norm_global)

        # Update dataset with normalized value
        first = 0
        normalized_sequences = []
        for length in self.lengths():
            last = first + length
            normalized_sequences.append(result_data[first:last])
            first = last
        self.data = np.asarray(normalized_sequences)

    @check_initialization
    def unnormalize(self):
        """Unnormalize dataset to original from zero mean and one std."""

        assert self.data is not None, \
            'Data field should be initialized: self.data = %s' % self.data

        if not self.normalized:
            return

        result_data = self._unnormalize_implementation(self.concatenated_data(),
                                                       self.data_shape[1:])
        # Update dataset with unnormalized value
        first = 0
        unnormalized_sequences = []
        for length in self.lengths():
            last = first + length
            unnormalized_sequences.append(result_data[first:last])
            first = last
        self.data = np.asarray(unnormalized_sequences)

    @staticmethod
    def padding(sequences, max_sequence_length, padding_value=0):
        """Add padding to data.

        Arguments:
            sequences                   Input sequences.
            max_sequence_length         Maximal sequence length.
            padding_value               Padding fill value.

        Return:
            data_with_padding           Output data with shape
                                        [len(sequences), max_sequence_length, ...].
            lengths                     Sequences lengths without padding.

        """
        sequences_with_padding = []
        lengths = []
        for sequence in sequences:
            if len(sequence) >= max_sequence_length:
                length = max_sequence_length
                sequence_with_padding = sequence[:max_sequence_length]
            else:
                length = len(sequence)
                shape = [max_sequence_length] + list(np.asarray(sequence).shape[1:])
                sequence_with_padding = np.ones(shape) * padding_value
                sequence_with_padding[:length] = sequence
            lengths.append(length)
            sequences_with_padding.append(sequence_with_padding)
        return np.asarray(sequences_with_padding), np.asarray(lengths)
