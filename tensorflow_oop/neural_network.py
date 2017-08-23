import tensorflow as tf
import numpy as np
import os
import time
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import *

# Set logging level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_inputs_values(function):
    """Decorator for check corresponding inputs values."""
    def wrapper(self, inputs_values, *args, **kwargs):
        assert np.all(np.asarray(inputs_values.shape[1:]) == np.asarray(self.inputs_shape_)), \
            'Inputs values shape should be correspond to model inputs shape: inputs_values.shape = %s, self.inputs_shape_ = %s' % (inputs_values.shape, self.inputs_shape_) 
        return function(self, inputs_values=inputs_values, *args, **kwargs)
    return wrapper

class TFNeuralNetwork(object):

    """
    Basic neural network model.
    """

    __slots__ = ['log_dir_', 'inputs_shape_', 'outputs_shape_',
                 'data_placeholder_', 'labels_placeholder_',
                 'outputs_', 'metrics_', 'loss_',
                 'sess_', 'kwargs_',
                 'summary_', 'summary_writer_', 'projector_config_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, inputs_type=tf.float32, outputs_type=tf.float32, reset_default_graph=True, metric_functions={}, **kwargs):
        print('Start initializing model...')

        # TensorBoard logging directory.
        self.log_dir_ = log_dir
        if tf.gfile.Exists(self.log_dir_):
            tf.gfile.DeleteRecursively(self.log_dir_)
        tf.gfile.MakeDirs(self.log_dir_)

        # Arguments.
        self.kwargs_ = kwargs

        # Reset default graph.
        if reset_default_graph:
            tf.reset_default_graph()

        # Input and Output layer shapes.
        self.inputs_shape_ = list(inputs_shape)
        self.outputs_shape_ = list(outputs_shape)

        # Generate placeholders for the data and labels.
        self.data_placeholder_ = tf.placeholder(inputs_type, shape=[None] + self.inputs_shape_, name='input_data')
        if not hasattr(self, 'labels_placeholder_'):
            self.labels_placeholder_ = tf.placeholder(outputs_type, shape=[None] + self.outputs_shape_, name='input_labels')

        # Build a Graph that computes predictions from the inference model.
        self.outputs_ = tf.identity(self.inference(self.data_placeholder_, **self.kwargs_), name='output_layer')

        # Loss function.
        self.loss_ = tf.identity(self.loss_function(self.outputs_, self.labels_placeholder_, **self.kwargs_), name='loss')

        # Evaluation options.
        self.metrics_ = {key : metric_functions[key](self.outputs_, self.labels_placeholder_) for key in metric_functions}
        self.metrics_['loss'] = self.loss_

        # Build the summary Tensor based on the TF collection of Summaries.
        for key in self.metrics_:
            tf.summary.scalar(key, self.metrics_[key])
        self.summary_ = tf.summary.merge_all()

        # Create a session for running Ops on the Graph.
        self.sess_ = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_ = tf.summary.FileWriter(self.log_dir_, self.sess_.graph)

        # Projector config object.
        self.projector_config_ = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

        # And then after everything is built:

        # Run the Op to initialize the variables.
        self.sess_.run(tf.global_variables_initializer())

        print('Finish initializing model.')

    def inference(self, inputs, **kwargs):
        """Model inference."""
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, outputs, labels_placeholder):
        """Loss function."""
        raise Exception('Loss function should be overwritten!')
        return loss

    def fit(self, train_set, iteration_count,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            epoch_count=None,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000):
        """Train model."""

        if epoch_count is not None:
            assert epoch_count > 0, \
                'Epoch count should be greater than zero: epoch_count = %s' % epoch_count
            epoch_count = int(epoch_count)
        else:
            assert iteration_count is not None, \
                'Iteration count should be passed if epoch count is None: iteration_count = %s, epoch_count = %s' % (iteration_count, epoch_count)
        if iteration_count is not None:
            assert iteration_count > 0, \
                'Iteration count should be greater than zero: iteration_count = %s' % iteration_count
            iteration_count = int(iteration_count)
        if summarizing_period is not None:
            assert summarizing_period > 0, \
                'Summarizing period should be greater than zero: summarizing_period = %s' % summarizing_period
            summarizing_period = int(summarizing_period)
        if logging_period is not None:
            assert logging_period > 0, \
                'Logging period should be greater than zero: logging_period = %s' % logging_period
            logging_period = int(logging_period)
        if checkpoint_period is not None:
            assert checkpoint_period > 0, \
                'Checkpoint period should be greater than zero: checkpoint_period = %s' % checkpoint_period
            checkpoint_period = int(checkpoint_period)
        if evaluation_period is not None:
            assert evaluation_period > 0, \
                'Evaluation period should be greater than zero: evaluation_period = %s' % evaluation_period
            evaluation_period = int(evaluation_period)
        assert isinstance(train_set, TFDataset), \
            'Training set should be object of TFDataset type: type(train_set) = %s' % type(train_set)
        assert train_set.init_, \
            'Training set should be initialized: train_set.init_ = %s' % train_set.init_
        if val_set is not None:
            assert(isinstance(val_set, TFDataset)), \
                'Validation set should be object of TFDataset type: type(val_set) = %s' % type(val_set)
            assert val_set.init_, \
                'Validation set should be initialized: val_set.init_ = %s' % val_set.init_

        print('Start training iteration...')
        start_fit_time = time.time()

        # Checkpoint configuration.
        checkpoint_name = 'fit-checkpoint'
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = optimizer(learning_rate).minimize(self.loss_)

        # Run the Op to initialize the variables.
        self.sess_.run(tf.global_variables_initializer())

        # Get actual iteration and epoch count
        if epoch_count is not None:
            iteration_count_by_epoch = train_set.size_ * epoch_count // train_set.batch_size_ + (1 if train_set.size_ % train_set.batch_size_ != 0 else 0)
            if iteration_count is not None:
                iteration_count = min(iteration_count, iteration_count_by_epoch)
            else:
                iteration_count = iteration_count_by_epoch
        else:
            epoch_count = iteration_count * train_set.batch_size_ // train_set.size_

        # Global iteration step and epoch number.
        iteration = 0
        epoch = 0

        # Start the training loop.
        iteration_times = []
        start_iteration_time = time.time()
        last_logging_iter = 0

        # Loop over all batches
        for batch in train_set.iterbatches(iteration_count):
            # Fill feed dict.
            feed_dict = {
                self.data_placeholder_: batch.data_,
                self.labels_placeholder_: batch.labels_,
            }

            # Run one step of the model training.
            self.sess_.run(train_op, feed_dict=feed_dict)

            # Save iteration time.
            iteration_times.append(time.time() - start_iteration_time)

            # Get current trained iteration and epoch.
            iteration += 1
            epoch = iteration * train_set.batch_size_ // train_set.size_

            # Write the summaries periodically.
            if iteration % summarizing_period == 0 or iteration == iteration_count:
                # Update the events file.
                summary_str = self.sess_.run(self.summary_, feed_dict=feed_dict)
                self.summary_writer_.add_summary(summary_str, iteration)

            # Print an overview periodically.
            if iteration % logging_period == 0 or iteration == iteration_count:
                # Calculate time of last period.
                duration = np.sum(iteration_times[-logging_period:])

                # Print logging info.
                metrics = self.evaluate(batch)
                metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                print('Iteration %d / %d (epoch %d / %d):   %s   [%.3f sec]' % (iteration, iteration_count, epoch, epoch_count, metrics_string, duration))

            # Save a checkpoint the model periodically.
            if checkpoint_period is not None and iteration % checkpoint_period == 0 or iteration == iteration_count:
                print('Saving checkpoint...')
                self.save(checkpoint_file, global_step=iteration)

            # Evaluate the model periodically.
            if evaluation_period is not None and iteration % evaluation_period == 0 or iteration == iteration_count:
                print('Evaluation...')
                start_evaluation_time = time.time()
                metrics = self.evaluate(train_set)
                metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                duration = time.time() - start_evaluation_time
                print('Evaluation on full dataset:   [training   set]   %s   [%.3f sec]' % (metrics_string, duration))
                if val_set is not None:
                    start_evaluation_time = time.time()
                    metrics = self.evaluate(val_set)
                    metrics_string = '   '.join([str(key) + ' = %.6f' % metrics[key] for key in metrics])
                    duration = time.time() - start_evaluation_time
                    print('Evaluation on full dataset:   [validation set]   %s   [%.3f sec]' % (metrics_string, duration))

            start_iteration_time = time.time()

        self.summary_writer_.flush()
        total_time = time.time() - start_fit_time
        print('Finish training iteration (total time %.3f sec).\n' % total_time)

    def evaluate(self, dataset):
        """Evaluate model."""
        assert isinstance(dataset, TFDataset) or isinstance(dataset, TFBatch), \
            'Argument should be object of TFDataset or TFBatch type: type(dataset) = %s' % type(dataset)
        if isinstance(dataset, TFDataset):
            assert dataset.init_, \
                'Dataset should be initialized: dataset.init_ = %s' % dataset.init_
        if isinstance(dataset, TFBatch):
            assert hasattr(dataset, 'data_') and hasattr(dataset, 'labels_'), \
                'Batch should contain attributes \'data_\' and \'labels_\'.'
        result = {}
        if len(self.metrics_) > 0:
            metric_keys = list(self.metrics_.keys())
            metric_values = list(self.metrics_.values())
            estimates = self.sess_.run(metric_values, feed_dict={
                self.data_placeholder_: dataset.data_,
                self.labels_placeholder_: dataset.labels_,
            })
            for i in range(len(self.metrics_)):
                result[metric_keys[i]] = estimates[i]
        return result

    def save(self, filename, global_step=None):
        """Save checkpoint."""
        saver = tf.train.Saver(max_to_keep=None)
        saved_filename = saver.save(self.sess_, filename, global_step=global_step)
        print('Model saved to: %s' % saved_filename)

    def load(self, model_checkpoint_path=None):
        """Load checkpoint."""
        if model_checkpoint_path is None:
            model_checkpoint_path = tf.train.latest_checkpoint(self.log_dir_)
            assert model_checkpoint_path is not None, \
                'Checkpoint path automatically not found.'
        saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta', clear_devices=True)
        saver.restore(self.sess_, model_checkpoint_path)
        print('Model loaded from: %s' % model_checkpoint_path)

    @check_inputs_values
    def forward(self, inputs_values):
        """Forward propagation."""
        return self.sess_.run(self.outputs_, feed_dict={
            self.data_placeholder_: inputs_values,
        })

    @check_inputs_values
    def top_k(self, inputs_values, k):
        """Top k element."""
        return self.sess_.run(tf.nn.top_k(self.data_placeholder_, k=k), feed_dict={
            self.data_placeholder_: inputs_values,
        })

    def __str__(self):
        string = 'TFNeuralNetwork object:\n'
        for attr in self.__slots__:
            string += "%20s: %s\n" % (attr, getattr(self, attr))
        return string[:-1]
