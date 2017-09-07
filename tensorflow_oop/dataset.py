"""
Dataset and batch base classes.
"""

import numpy as np
import pickle

def check_initialization(function):
    """Decorator for check initialization."""
    def wrapper(self, *args, **kwargs):
        assert self.init, \
            'Object should be initialized: self.init = %s' % self.init
        return function(self, *args, **kwargs)
    return wrapper

class TFBatch:

    """
    Batch container.
    """

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        string = "TFBatch object:\n"
        for attr in self.__dict__:
            string += "%s: \n%s\n" % (attr, getattr(self, attr))
        return string[:-1]

class TFDataset(object):

    """
    Dataset structure.
    """

    __slots__ = ['init', 'size',
                 'data', 'data_shape', 'data_ndim',
                 'labels', 'labels_shape', 'labels_ndim',
                 'batch_size', 'batch_num',
                 'normalized', 'normalization_mask',
                 'normalization_mean', 'normalization_std']

    def __init__(self, data=None, labels=None):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.init = False
        self.size = 0
        self.batch_size = 1
        self.batch_num = 0
        if data is not None or labels is not None:
            self.initialize(data=data, labels=labels)

    def copy(self, other):
        """Copy other dataframe."""
        for attr in self.__slots__:
            setattr(self, attr, getattr(other, attr))

    def initialize(self, data, labels):
        """Set data and labels."""
        assert data is not None or labels is not None, \
            '''Data or labels should be passed:
            data = %s, labels = %s''' % (data, labels)

        if data is not None and labels is not None:
            assert len(data) == len(labels), \
                '''Data and labels should be the same length:
                len(data) = %s, len(labels) = %s''' % (len(data), len(labels))

        # Processing data
        if data is not None:
            self.size = len(data)
            data = np.asarray(data)
            if data.ndim == 1:
                self.data = np.reshape(data, (self.size, 1))
            else:
                self.data = data
            self.data_shape = list(self.data.shape[1:])
            self.data_ndim = len(self.data_shape)
        else:
            self.data = None
            self.data_shape = None
            self.data_ndim = None
            self.normalized = None
            self.normalization_mask = None
            self.normalization_mean = None
            self.normalization_std = None

        # Processing labels
        if labels is not None:
            self.size = len(labels)
            labels = np.asarray(labels)
            if labels.ndim == 1:
                self.labels = np.reshape(labels, (self.size, 1))
            else:
                self.labels = labels
            self.labels_shape = list(self.labels.shape[1:])
            self.labels_ndim = len(self.labels_shape)
        else:
            self.labels = None
            self.labels_shape = None
            self.labels_ndim = None

        self.init = True

    @check_initialization
    def shuffle(self):
        """Random shuffling of dataset."""
        indexes = np.arange(self.size)
        np.random.shuffle(indexes)
        self.data = self.data[indexes]
        self.labels = self.labels[indexes]

    @check_initialization
    def set_batch_size(self, batch_size):
        """Set batch size."""
        assert batch_size > 0, \
            '''Batch size should be greater then zero:
            batch_size = %s''' % batch_size

        assert batch_size <=  self.size, \
            '''Batch size should not be greater then dataset size:
            batch_size = %s, self.size = %s''' % (batch_size, self.size)
        self.batch_size = int(batch_size)

    @check_initialization
    def next_batch(self):
        """Get next batch."""
        first = (self.batch_num * self.batch_size) % self.size
        last = first + self.batch_size
        batch_data = None
        batch_labels = None
        if (last <= self.size):
            if self.data is not None:
                batch_data = self.data[first:last]
            if self.labels is not None:
                batch_labels = self.labels[first:last]
        else:
            if self.data is not None:
                batch_data = np.append(self.data[first:],
                                       self.data[:last - self.size],
                                       axis=0)
            if self.labels is not None:
                batch_labels = np.append(self.labels[first:],
                                         self.labels[:last - self.size],
                                         axis=0)
        self.batch_num += 1
        return TFBatch(data=batch_data, labels=batch_labels)

    @check_initialization
    def iterbatches(self, count=None):
        """Get iterator by batches."""
        if count is None:
            count = self.size // self.batch_size
            if self.size % self.batch_size != 0:
                count += 1
        for i in range(count):
            yield self.next_batch()

    @check_initialization
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set."""
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
            train_set = TFDataset()
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
            val_set = TFDataset()
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
            test_set = TFDataset()
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
        """Load dataset from file."""
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        assert isinstance(obj, TFDataset), \
            '''Loaded object should be TFDataset object:
            type(obj) = %s''' % type(obj)
        return obj

    @check_initialization
    def save(self, filename):
        """Save dataset to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=-1)

    @check_initialization
    def generate_sequences(self,
                           sequence_length,
                           sequence_step,
                           label_length=None,
                           label_offset=None):
        """Generate sequences."""
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

            # Create dataset
            dataset = TFDataset(sequences, labels)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=labels)
        else:
            sequences = []
            last = self.size - sequence_length + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data[i : last_sequence_index]
                sequences.append(current_sequence)
            
            # Create dataset
            dataset = TFDataset(sequences)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=None)

        # Update normalization mask
        if self.normalization_mask is not None:
            dataset.normalization_mask = [False] + self.normalization_mask

        return dataset

    @check_initialization
    def normalize(self, mask=None):
        """
        Normalize data to zero mean and one std by mask.
        Where mask is boolean indicators corresponding to data dimensions.
        If mask value is True, then feature with this dimension should be normalized.
        """
        assert self.data is not None, \
            '''Data field should be initialized: self.data = %s''' % self.data

        if self.normalized:
            return

        if mask is not None:
            assert len(mask) == self.data_ndim, \
                '''Mask length should be equal to data dimensions count:
                len(mask) = %s, self.data_ndim = %s''' \
                % (len(mask), self.data_ndim)

            for i in range(0, len(mask) - 1):
                assert mask[i + 1] or not mask[i], \
                    '''False elements should be before True elements:
                    mask = %s''' % mask

            assert mask[-1] == True, \
                '''Last mask element should be True:
                mask = %s''' % mask

            # Reshape to array of features
            data_shapearr = np.asarray(self.data_shape)
            new_shape = [-1] + list(data_shapearr[mask])
            reshaped_data = np.reshape(self.data, new_shape)

            # Save normalisation properties
            self.normalization_mask = list(mask)
            self.normalization_mean = np.mean(reshaped_data, axis=0)
            self.normalization_std = np.std(reshaped_data, axis=0)

            # Reshape normalization properties for correct broadcasting
            valid_shape = data_shapearr
            valid_shape[np.logical_not(self.normalization_mask)] = 1
            reshaped_mean = np.reshape(self.normalization_mean, valid_shape)
            reshaped_std = np.reshape(self.normalization_std, valid_shape)

            # Replace zero std with one
            valid_std = reshaped_std
            valid_std[reshaped_std == 0] = 1

            # Update dataset with normalized value
            self.data = (self.data - reshaped_mean) / valid_std
        else:
            # Save normalisation properties
            self.normalization_mask = None
            self.normalization_mean = np.mean(self.data)
            self.normalization_std = np.std(self.data)

            # Update dataset with normalized value
            self.data = (self.data - self.normalization_mean) / self.normalization_std

        self.normalized = True

    @check_initialization
    def unnormalize(self):
        """Unnormalize dataset to original from zero mean and one std."""
        assert self.data is not None, \
            'Data field should be initialized: self.data = %s' % self.data

        if not self.normalized:
            return

        if self.normalization_mask is not None:
            data_shapearr = np.asarray(self.data_shape)

            # Reshape for correct broadcasting
            valid_shape = data_shapearr
            valid_shape[np.logical_not(self.normalization_mask)] = 1
            reshaped_mean = np.reshape(self.normalization_mean, valid_shape)
            reshaped_std = np.reshape(self.normalization_std, valid_shape)

            # Replace zero std with one 
            valid_std = reshaped_std
            valid_std[reshaped_std == 0] = 1

            # Update dataset with unnormalized value
            self.data = self.data*valid_std +  reshaped_mean
        else:
            # Update dataset with unnormalized value
            self.data =  self.data*self.normalization_std + self.normalization_mean

        self.normalized = False

    @check_initialization
    def one_hot(self, encoding_size, dtype=np.float):
        """One hot encoding."""
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
        return '%s : %s -> %s' % (self.size, self.data_shape, self.labels_shape)

    def __len__(self):
        return self.size

    def __str__(self):
        string = 'TFDataset object:\n'
        for attr in self.__slots__:
            if attr != 'data' and attr != 'labels':
                string += "%20s: %s\n" % (attr, getattr(self, attr))
        if 'data' in self.__slots__:
            string += "%s: \n%s\n" % ('data', getattr(self, 'data'))
        if 'labels' in self.__slots__:
            string += "%s: \n%s\n" % ('labels', getattr(self, 'labels'))
        return string[:-1]
