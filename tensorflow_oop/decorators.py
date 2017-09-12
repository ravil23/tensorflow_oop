"""
Decorators.
"""

import tensorflow as tf
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import *


def check_inputs_values(function):
    """Decorator for check corresponding inputs values."""
    def wrapper(self, inputs_values, *args, **kwargs):
        new_shape = np.asarray(np.asarray(inputs_values).shape[1:])
        cur_shape = np.asarray(self.inputs_shape)
        assert np.all(new_shape == cur_shape), \
            '''Inputs values shape should be correspond to model inputs shape:
            inputs_values.shape = %s, self.inputs_shape = %s''' \
            % (inputs_values.shape, self.inputs_shape) 
        return function(self, inputs_values=inputs_values, *args, **kwargs)
    return wrapper


def check_fit_arguments(function):
    """Decorator for check fit arguments."""
    def wrapper(self,
                train_set,
                epoch_count=None,
                iter_count=None,
                optimizer=tf.train.RMSPropOptimizer,
                learning_rate=0.001,
                val_set=None,
                summarizing_period=100,
                logging_period=100,
                checkpoint_period=10000,
                evaluation_period=10000,
                max_gradient_norm=None,
                *args,
                **kwargs):
        assert isinstance(learning_rate, tf.Tensor) or learning_rate > 0, \
            '''Learning rate should be tensor or greater than zero:
            learning_rate = %s''' % learning_rate

        assert epoch_count is not None or iter_count is not None, \
            '''Epoch or iter count should be passed:
            epoch_count = %s, iter_count = %s''' \
            % (epoch_count, iter_count)

        if epoch_count is not None:
            assert epoch_count > 0, \
                '''Epoch count should be greater than zero:
                epoch_count = %s''' % epoch_count
            epoch_count = int(epoch_count)
        else:
            assert iter_count is not None, \
                '''Iteration count should be passed if epoch count is None:
                iter_count = %s, epoch_count = %s''' \
                % (iter_count, epoch_count)

        if iter_count is not None:
            assert iter_count > 0, \
                '''Iteration count should be greater than zero:
                iter_count = %s''' % iter_count
            iter_count = int(iter_count)

        if summarizing_period is not None:
            assert summarizing_period > 0, \
                '''Summarizing period should be greater than zero:
                summarizing_period = %s''' % summarizing_period
            summarizing_period = int(summarizing_period)

        if logging_period is not None:
            assert logging_period > 0, \
                '''Logging period should be greater than zero:
                logging_period = %s''' % logging_period
            logging_period = int(logging_period)

        if checkpoint_period is not None:
            assert checkpoint_period > 0, \
                '''Checkpoint period should be greater than zero:
                checkpoint_period = %s''' % checkpoint_period
            checkpoint_period = int(checkpoint_period)

        if evaluation_period is not None:
            assert evaluation_period > 0, \
                '''Evaluation period should be greater than zero:
                evaluation_period = %s''' % evaluation_period
            evaluation_period = int(evaluation_period)

        assert isinstance(train_set, TFDataset), \
            '''Training set should be object of TFDataset type:
            type(train_set) = %s''' % type(train_set)

        assert train_set.init, \
            '''Training set should be initialized:
            train_set.init = %s''' % train_set.init

        if val_set is not None:
            assert(isinstance(val_set, TFDataset)), \
                '''Validation set should be object of TFDataset type:
                type(val_set) = %s''' % type(val_set)

            assert val_set.init, \
                '''Validation set should be initialized:
                val_set.init = %s''' % val_set.init

        return function(self, 
                        train_set,
                        epoch_count,
                        iter_count,
                        optimizer,
                        learning_rate,
                        val_set,
                        summarizing_period,
                        logging_period,
                        checkpoint_period,
                        evaluation_period,
                        max_gradient_norm,
                        *args,
                        **kwargs)
    return wrapper


def check_evaluate_arguments(function):
    """Decorator for check fit arguments."""
    def wrapper(self, data, collection='eval_test', iteration=None, *args, **kwargs):
        assert collection in ['batch_train',
                              'batch_validation',
                              'log_train',
                              'eval_train',
                              'eval_validation',
                              'eval_test'], \
            '''Collections should be only from list
            ['batch_train',
             'batch_validation',
             'log_train',
             'eval_train',
             'eval_validation',
             'eval_test']:
            collection = %s''' % collection

        assert isinstance(data, TFDataset) or isinstance(data, TFBatch), \
            '''Argument should be object of TFDataset or TFBatch type:
            type(data) = %s''' % type(data)

        if isinstance(data, TFDataset):
            assert data.init, \
                '''Dataset should be initialized:
                data.init = %s''' % data.init

        if isinstance(data, TFBatch):
            assert hasattr(data, 'data') and hasattr(data, 'labels'), \
                '''Batch should contain attributes \'data\' and \'labels\'.'''

        return function(self, data, collection, iteration, *args, **kwargs)
    return wrapper


def check_triplets_data_labels(function):
    """Decorator for check triplets data and labels."""
    def wrapper(self, data, labels):
        assert data is not None and labels is not None, \
            '''Data and labels should be passed:
            data = %s, labels = %s''' % (data, labels)

        len_flatten_labels = len(np.asarray(labels).flatten())
        len_labels = len(labels)
        assert len_flatten_labels == len_labels, \
            '''Flatten labels should be the same length:
            len(flatten_labels) = %s, len(labels) = %s''' % \
            (len_flatten_labels, len_labels)

        assert len(data) == len(labels), \
            '''Data and labels should be the same length:
            len(data) = %s, len(labels) = %s''' % (len(data), len(labels))
        return function(self, data, labels)
    return wrapper
