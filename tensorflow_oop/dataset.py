import numpy as np
import pickle

def check_initialization(function):
    """Decorator for check initialization."""
    def wrapper(self, *args, **kwargs):
        assert self.init_, \
            'Object should be initialized: self.init_ = %s' % self.init_
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

class TFDataset:

    """
    Dataset structure.
    """

    __slots__ = ['init_', 'size_',
                 'data_', 'data_shape_', 'data_ndim_',
                 'labels_', 'labels_shape_', 'labels_ndim_',
                 'batch_size_', 'batch_num_',
                 'normalized_', 'normalization_mask_', 'normalization_mean_', 'normalization_std_']

    def __init__(self, data=None, labels=None):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.init_ = False
        self.size_ = 0
        self.batch_size_ = 1
        self.batch_num_ = 0
        if data is not None or labels is not None:
            self.initialize(data=data, labels=labels)

    def copy(self, other):
        """Copy other dataframe."""
        for attr in self.__slots__:
            setattr(self, attr, getattr(other, attr))

    def initialize(self, data, labels):
        """Set data and labels."""
        assert data is not None or labels is not None, \
            'Data or labels should be passed: data = %s, labels = %s' % (data, labels)
        if data is not None and labels is not None:
            assert len(data) == len(labels), \
                'Data and labels should be the same length: len(data) = %s, len(labels) = %s' % (len(data), len(labels))
        if data is not None:
            self.size_ = len(data)
            data = np.asarray(data)
            if data.ndim == 1:
                self.data_ = np.reshape(data, (self.size_, 1))
            else:
                self.data_ = data
            self.data_shape_ = list(self.data_.shape[1:])
            self.data_ndim_ = len(self.data_shape_)
        else:
            self.data_ = None
            self.data_shape_ = None
            self.data_ndim_ = None
            self.normalized_ = None
            self.normalization_mask_ = None
            self.normalization_mean_ = None
            self.normalization_std_ = None
        if labels is not None:
            self.size_ = len(labels)
            labels = np.asarray(labels)
            if labels.ndim == 1:
                self.labels_ = np.reshape(labels, (self.size_, 1))
            else:
                self.labels_ = labels
            self.labels_shape_ = list(self.labels_.shape[1:])
            self.labels_ndim_ = len(self.labels_shape_)
        else:
            self.labels_ = None
            self.labels_shape_ = None
            self.labels_ndim_ = None
        self.init_ = True

    @check_initialization
    def shuffle(self):
        """Random shuffling of dataset."""
        indexes = np.arange(self.size_)
        np.random.shuffle(indexes)
        self.data_ = self.data_[indexes]
        self.labels_ = self.labels_[indexes]

    @check_initialization
    def set_batch_size(self, batch_size):
        """Set batch size."""
        assert batch_size > 0, \
            'Batch size should be greater then zero: batch_size = %s' % batch_size
        assert batch_size <=  self.size_, \
            'Batch size should not be greater then dataset size: batch_size = %s, self.size_ = %s' % (batch_size, self.size_)
        self.batch_size_ = int(batch_size)

    @check_initialization
    def next_batch(self):
        """Get next batch."""
        first = (self.batch_num_ * self.batch_size_) % self.size_
        last = first + self.batch_size_
        batch_data = None
        batch_labels = None
        if (last <= self.size_):
            if self.data_ is not None:
                batch_data = self.data_[first:last]
            if self.labels_ is not None:
                batch_labels = self.labels_[first:last]
        else:
            if self.data_ is not None:
                batch_data = np.append(self.data_[first:], self.data_[:last - self.size_], axis=0)
            if self.labels_ is not None:
                batch_labels = np.append(self.labels_[first:], self.labels_[:last - self.size_], axis=0)
        self.batch_num_ += 1
        return TFBatch(data_=batch_data, labels_=batch_labels)

    @check_initialization
    def iterbatches(self, count=None):
        """Get iterator by batches."""
        if count is None:
            count = self.size_ // self.batch_size_ + (1 if self.size_ % self.batch_size_ != 0 else 0)
        for i in range(count):
            yield self.next_batch()

    @check_initialization
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set."""
        assert train_size >= 0, \
            'Training size should not be less then zero: train_size = %s' % train_size
        assert val_size >= 0, \
            'Validation size should not be less then zero: val_size = %s' % val_size
        assert test_size >= 0, \
            'Testing size should not be less then zero: test_size = %s' % test_size
        total_size = train_size + val_size + test_size
        assert total_size == self.size_ or total_size == 1, \
            'Total size should be equal to TFDataset size or one: total_size = %s, self.size_ = %s' % (total_size, self.size_)
        if total_size == 1:
            if train_size != 0:
                train_size = int(round(float(train_size) * self.size_))
            if test_size != 0:
                if val_size != 0:
                    test_size = int(round(float(test_size) * self.size_))
                else:
                    test_size = self.size_ - train_size
            if val_size != 0:
                val_size = self.size_ - train_size - test_size
        indexes = np.arange(self.size_)
        if shuffle:
            np.random.shuffle(indexes)
        if train_size > 0:
            train_set = TFDataset()
            train_set.copy(self)
            data = self.data_[indexes[:train_size]] if self.data_ is not None else None
            labels = self.labels_[indexes[:train_size]] if self.labels_ is not None else None
            train_set.initialize(data, labels)
        else:
            train_set = None
        if val_size > 0:
            val_set = TFDataset()
            val_set.copy(self)
            data = self.data_[indexes[train_size:train_size + val_size]] if self.data_ is not None else None
            labels = self.labels_[indexes[train_size:train_size + val_size]] if self.labels_ is not None else None
            val_set.initialize(data, labels)
        else:
            val_set = None
        if test_size > 0:
            test_set = TFDataset()
            test_set.copy(self)
            data = self.data_[indexes[-test_size:]] if self.data_ is not None else None
            labels = self.labels_[indexes[-test_size:]] if self.labels_ is not None else None
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
            'Loaded object should be TFDataset object: type(obj) = %s' % type(obj)
        return obj

    @check_initialization
    def save(self, filename):
        """Save dataset to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @check_initialization
    def generate_sequences(self, sequence_length, sequence_step, label_length=None, label_offset=None):
        """Generate sequences."""
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        assert sequence_length > 0, \
            'Sequence length should be greater than zero: sequence_length = %s' % sequence_length
        assert sequence_step > 0, \
            'Sequence step should be greater than zero: sequence_step = %s' % sequence_step
        if label_length is not None:
            assert label_length > 0 and label_offset is not None, \
                'Label length should be greater than zero and label offset passed: label_length = %s, label_offset = %s' % (label_length, label_offset)
        if label_length is not None:
            sequences = []
            labels = []
            last = self.size_ - sequence_length - label_length - label_offset + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data_[i : last_sequence_index]
                sequences.append(current_sequence)
                first_label_index = last_sequence_index + label_offset
                current_label = self.data_[first_label_index : first_label_index + label_length]
                labels.append(current_label)
            dataset = TFDataset(sequences, labels)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=labels)
        else:
            sequences = []
            last = self.size_ - sequence_length + 1
            for i in range(0, last, sequence_step):
                last_sequence_index = i + sequence_length
                current_sequence = self.data_[i : last_sequence_index]
                sequences.append(current_sequence)
            dataset = TFDataset(sequences)
            dataset.copy(self)
            dataset.initialize(data=sequences, labels=None)
        if self.normalization_mask_ is not None:
            dataset.normalization_mask_ = [False] + self.normalization_mask_
        return dataset

    @check_initialization
    def normalize(self, mask=None):
        """
        Normalize data to zero mean and one std by mask.
        Where mask is boolean indicators corresponding to data dimensions.
        If mask value is True, then feature with this dimension should be normalized.
        """
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        if self.normalized_:
            return
        if mask is not None:
            assert len(mask) == self.data_ndim_, \
                'Mask length should be equal to data dimensions count: len(mask) = %s, self.data_ndim_ = %s' % (len(mask), self.data_ndim_)

            for i in range(0, len(mask) - 1):
                assert mask[i + 1] or not mask[i], \
                    'False elements should be before True elements: mask = %s' % mask

            assert mask[-1] == True, \
                'Last mask element should be True: mask = %s' % mask

            # Reshape to array of features
            data_shape_arr = np.asarray(self.data_shape_)
            new_shape = [-1] + list(data_shape_arr[mask])
            reshaped_data = np.reshape(self.data_, new_shape)

            # Save normalisation properties
            self.normalization_mask_ = list(mask)
            self.normalization_mean_ = np.mean(reshaped_data, axis=0)
            self.normalization_std_ = np.std(reshaped_data, axis=0)

            # Reshape normalization properties for correct broadcasting
            valid_shape = data_shape_arr
            valid_shape[np.logical_not(self.normalization_mask_)] = 1
            reshaped_normalization_mean_ = np.reshape(self.normalization_mean_, valid_shape)
            reshaped_normalization_std_ = np.reshape(self.normalization_std_, valid_shape)

            # Replace zero std with one
            valid_normalization_std_ = reshaped_normalization_std_
            valid_normalization_std_[reshaped_normalization_std_ == 0] = 1

            # Update dataset with normalized value
            self.data_ = (self.data_ - reshaped_normalization_mean_) / valid_normalization_std_
        else:
            # Save normalisation properties
            self.normalization_mask_ = None
            self.normalization_mean_ = np.mean(self.data_)
            self.normalization_std_ = np.std(self.data_)

            # Update dataset with normalized value
            self.data_ = (self.data_ - self.normalization_mean_) / self.normalization_std_
        self.normalized_ = True

    @check_initialization
    def unnormalize(self):
        """Unnormalize dataset to original from zero mean and one std."""
        assert self.data_ is not None, \
            'Data field should be initialized: self.data_ = %s' % self.data_
        if not self.normalized_:
            return
        if self.normalization_mask_ is not None:
            data_shape_arr = np.asarray(self.data_shape_)

            # Reshape for correct broadcasting
            valid_shape = data_shape_arr
            valid_shape[np.logical_not(self.normalization_mask_)] = 1
            reshaped_normalization_mean_ = np.reshape(self.normalization_mean_, valid_shape)
            reshaped_normalization_std_ = np.reshape(self.normalization_std_, valid_shape)

            # Replace zero std with one 
            valid_normalization_std_ = reshaped_normalization_std_
            valid_normalization_std_[reshaped_normalization_std_ == 0] = 1

            # Update dataset with unnormalized value
            self.data_ = self.data_ * valid_normalization_std_ +  reshaped_normalization_mean_
        else:
            # Update dataset with unnormalized value
            self.data_ =  self.data_ * self.normalization_std_ +  self.normalization_mean_
        self.normalized_ = False

    def __len__(self):
        return self.size_

    def __str__(self):
        string = 'TFDataset object:\n'
        for attr in self.__slots__:
            if attr != 'data_' and attr != 'labels_':
                string += "%20s: %s\n" % (attr, getattr(self, attr))
        if 'data_' in self.__slots__:
            string += "%s: \n%s\n" % ('data_', getattr(self, 'data_'))
        if 'labels_' in self.__slots__:
            string += "%s: \n%s\n" % ('labels_', getattr(self, 'labels_'))
        return string[:-1]
