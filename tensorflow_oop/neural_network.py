import tensorflow as tf
import numpy as np
import os
import time
import sys
from tensorflow.contrib.tensorboard.plugins import projector

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
        new_shape = np.asarray(inputs_values.shape[1:])
        cur_shape = np.asarray(self.inputs_shape)
        assert np.all(new_shape == cur_shape), \
            '''Inputs values shape should be correspond to model inputs shape:
            inputs_values.shape = %s, self.inputs_shape = %s''' \
            % (inputs_values.shape, self.inputs_shape) 
        return function(self, inputs_values=inputs_values, *args, **kwargs)
    return wrapper

class TFNeuralNetwork(object):

    """
    Basic neural network model.
    """

    __slots__ = ['init', 'log_dir',
                 'inputs_shape', 'targets_shape', 'outputs_shape',
                 'inputs', 'targets', 'outputs',
                 'metrics', 'loss',
                 'sess', 'kwargs',
                 'summary', 'summary_writer', 'projector_config']

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.init = False

    def load(self, model_checkpoint_path=None):
        """Load checkpoint."""
        if model_checkpoint_path is None:
            model_checkpoint_path = tf.train.latest_checkpoint(self.log_dir)
            assert model_checkpoint_path is not None, \
                'Checkpoint path automatically not found.'

        print('Start loading model...')

        # Get metagraph saver
        saver = tf.train.import_meta_graph(model_checkpoint_path + '.meta',
                                           clear_devices=True)

        # Create a session for running Ops on the Graph
        self.sess = tf.Session()

        # Restore model from saver
        saver.restore(self.sess, model_checkpoint_path)

        # Get named tensors
        self.inputs = self.sess.graph.get_tensor_by_name('inputs:0')
        self.targets = self.sess.graph.get_tensor_by_name('targets:0')
        self.outputs = self.sess.graph.get_tensor_by_name('outputs:0')
        self.loss = self.sess.graph.get_tensor_by_name('loss_function:0')

        # Input, Target and Output layer shapes
        self.inputs_shape = self.inputs.shape.as_list()[1:]
        self.targets_shape = self.targets.shape.as_list()[1:]
        self.outputs_shape = self.outputs.shape.as_list()[1:]

        # Instantiate a SummaryWriter to output summaries and the Graph
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)

        # Projector config object
        self.projector_config = projector.ProjectorConfig()

        # Enable initialization flag
        self.init = True

        print('Model loaded from: %s' % model_checkpoint_path)

    def initialize(self,
                   inputs_shape,
                   targets_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   targets_type=tf.float32,
                   outputs_type=tf.float32,
                   reset=True,
                   **kwargs):
        """Initialize model.

        Arguments:
            inputs_shape -- shape of inputs layer
            targets_shape -- shape of targets layer
            outputs_shape -- shape of outputs layer
            inputs_type -- type of inputs layer
            targets_type -- type of targets layer
            outputs_type -- type of outputs layer
            reset -- indicator of clearing default graph and logging directory
            kwargs -- dictionary of keyword arguments

        """
        print('Start initializing model...')

        # Reset if necessary
        if reset:
            tf.reset_default_graph()

            # Clean TensorBoard logging directory
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)

        # Create TensorBoard logging directory
        if not tf.gfile.Exists(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)

        # Arguments
        self.kwargs = kwargs

        # Input, Target and Output layer shapes
        self.inputs_shape = list(inputs_shape)
        self.targets_shape = list(targets_shape)
        self.outputs_shape = list(outputs_shape)

        # Generate placeholders for the inputs and targets
        self.inputs = tf.placeholder(inputs_type,
                                     shape=[None] + self.inputs_shape,
                                     name='inputs')
        if not hasattr(self, 'targets'):
            self.targets = tf.placeholder(targets_type,
                                          shape=[None] + self.targets_shape,
                                          name='targets')

        # Build a Graph that computes predictions from the inference model
        outputs = self.inference(self.inputs, **self.kwargs)
        self.outputs = tf.identity(outputs, name='outputs')

        # Loss function
        loss = self.loss_function(self.targets, self.outputs, **self.kwargs)
        self.loss = tf.identity(loss, name='loss_function')

        # Evaluation options
        self.metrics = {}
        self.metrics['loss'] = self.loss
        tf.summary.scalar('loss', self.loss)

        # Create a session for running Ops on the Graph
        self.sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph
        self.summary_writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)

        # Projector config object
        self.projector_config = projector.ProjectorConfig()

        # Run the Op to initialize the variables
        self.sess.run(tf.global_variables_initializer())

        # Enable initialization flag
        self.init = True

        print('Finish initializing model.')

    def add_metric(self, key, metric_function):
        """Add logging and summarizing metric.

        Arguments:
            key -- string name
            metric_function -- function object with input arguments in format
                function(targets, outputs)

        """
        assert isinstance(key, str), \
            '''Key should be string format:
            type(key) = %s''' % type(key)
        assert key not in self.metrics, \
            '''Key should be unique, but this key is exists:
            key = %s''' % key
        metric = metric_function(self.targets, self.outputs)
        if len(metric.shape) == 1 or \
           len(metric.shape) == 2 and metric.shape[1] == 1:
            tf.summary.scalar(key, metric)
        else:
            tf.summary.histogram(key, metric)
        self.metrics[key] = metric

    def inference(self, inputs, **kwargs):
        """Model inference.

        Arguments:
            inputs -- tensor of batch with inputs
            kwargs -- dictionary of keyword arguments

        Return:
            outputs -- tensor of outputs layer

        """
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, targets, outputs, **kwargs):
        """Loss function.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of keyword arguments

        Return:
            loss -- tensorflow operation for minimization

        """
        raise Exception('Loss function should be overwritten!')
        return loss

    def fit(self,
            train_set,
            epoch_count=None,
            iter_count=None,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000):
        """Train model.

        Arguments:
            train_set -- dataset for training
            epoch_count -- training epochs count
            iter_count -- training iterations count
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step
            val_set -- dataset for validation
            summarizing_period -- iterations count between summarizing
            logging_period -- iterations count between logging to stdout
            checkpoint_period -- iterations count between saving checkpoint
            evaluation_period -- iterations count between evaluation

        """
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
                '''iter count should be passed if epoch count is None:
                iter_count = %s, epoch_count = %s''' \
                % (iter_count, epoch_count)

        if iter_count is not None:
            assert iter_count > 0, \
                '''iter count should be greater than zero:
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

        print('Start training iter...')
        start_fit_time = time.time()

        # Checkpoint configuration
        checkpoint_name = 'fit-checkpoint'
        checkpoint_file = os.path.join(self.log_dir, checkpoint_name)

        # Add to the Graph the Ops that calculate and apply gradients
        train_op = optimizer(learning_rate).minimize(self.loss)

        # Run the Op to initialize the variables
        self.sess.run(tf.global_variables_initializer())

        # Get actual iter and epoch count
        if epoch_count is not None:
            iter_count_by_epoch = (train_set.size_ * epoch_count) // train_set.batch_size_
            if train_set.size_ % train_set.batch_size_ != 0:
                iter_count_by_epoch += 1
            if iter_count is not None:
                iter_count = min(iter_count, iter_count_by_epoch)
            else:
                iter_count = iter_count_by_epoch
        else:
            epoch_count = (iter_count * train_set.batch_size_) // train_set.size_

        # Global iter step and epoch number
        iteration = 0
        epoch = 0

        # Start the training loop
        iter_times = []
        start_iter_time = time.time()
        last_logging_iter = 0

        # Loop over all batches
        for batch in train_set.iterbatches(iter_count):
            # Fill feed dict
            feed_dict = {
                self.inputs: batch.data_,
                self.targets: batch.labels_,
            }

            # Run one step of the model training
            self.sess.run(train_op, feed_dict=feed_dict)

            # Save iter time
            iter_times.append(time.time() - start_iter_time)

            # Get current trained iter and epoch
            iteration += 1
            epoch = iteration * train_set.batch_size_ // train_set.size_

            # Write the summaries periodically
            if iteration % summarizing_period == 0 or iteration == iter_count:
                # Update the events file.
                summary_str = self.sess.run(tf.summary.merge_all(),
                                            feed_dict=feed_dict)
                self.summary_writer.add_summary(summary_str, iteration)

            # Print an overview periodically
            if iteration % logging_period == 0 or iteration == iter_count:
                # Calculate time of last period
                duration = np.sum(iter_times[-logging_period:])

                # Print logging info
                metrics = self.evaluate(batch)
                metrics_list = ['%s = %.6f' % (k, metrics[k]) for k in metrics]
                format_string = 'Iter %d / %d (epoch %d / %d):   %s   [%.3f sec]'
                print(format_string % (iteration, iter_count,
                                       epoch, epoch_count,
                                       '   '.join(metrics_list),
                                       duration))

            # Save a checkpoint the model periodically
            if (checkpoint_period is not None and \
               iteration % checkpoint_period == 0) or \
               iteration == iter_count:
                print('Saving checkpoint...')
                self.save(checkpoint_file, global_step=iteration)

            # Evaluate the model periodically
            if (evaluation_period is not None and \
               iteration % evaluation_period == 0) or \
               iteration == iter_count:
                print('Evaluation...')

                # Eval on training set
                start_evaluation_time = time.time()
                metrics = self.evaluate(train_set)
                metrics_list = ['%s = %.6f' % (k, metrics[k]) for k in metrics]
                duration = time.time() - start_evaluation_time
                format_string = 'Eval on [training   set]:   %s   [%.3f sec]'
                print(format_string % ('   '.join(metrics_list),
                                       duration))

                # Eval on validation set if necessary
                if val_set is not None:
                    start_evaluation_time = time.time()
                    metrics = self.evaluate(val_set)
                    metrics_list = ['%s = %.6f' % (k, metrics[k]) for k in metrics]
                    duration = time.time() - start_evaluation_time
                    format_string = 'Eval on [validation set]:   %s   [%.3f sec]'
                    print(format_string % ('   '.join(metrics_list),
                                           duration))

            start_iter_time = time.time()

        self.summary_writer.flush()
        total_time = time.time() - start_fit_time
        print('Finish training iteration (total time %.3f sec).\n' % total_time)

    def evaluate(self, data):
        """Evaluate model.

        Arguments:
            data -- batch or dataset of inputs

        Return:
            result -- metrics dictionary

        """
        assert isinstance(data, TFDataset) or isinstance(data, TFBatch), \
            '''Argument should be object of TFDataset or TFBatch type:
            type(data) = %s''' % type(data)

        if isinstance(data, TFDataset):
            assert data.init_, \
                '''Dataset should be initialized:
                data.init_ = %s''' % data.init_

        if isinstance(data, TFBatch):
            assert hasattr(data, 'data_') and hasattr(data, 'labels_'), \
                '''Batch should contain attributes \'data_\' and \'labels_\'.'''

        result = {}
        if len(self.metrics) > 0:
            metric_keys = list(self.metrics.keys())
            metric_values = list(self.metrics.values())
            estimates = self.sess.run(metric_values, feed_dict={
                self.inputs: dataset.data_,
                self.targets: dataset.labels_,
            })
            for i in range(len(self.metrics)):
                result[metric_keys[i]] = estimates[i]
        return result

    def save(self, filename, global_step=None):
        """Save checkpoint.

        Arguments:
            filename -- path to saving
            global_step -- optional suffix adding to path (default None)

        """
        saver = tf.train.Saver(max_to_keep=None)
        saved_filename = saver.save(self.sess, filename,
                                    global_step=global_step)
        print('Model saved to: %s' % saved_filename)

    @check_inputs_values
    def forward(self, inputs_values):
        """Forward propagation.

        Arguments:
            inputs_values -- batch of inputs

        Return:
            outputs_values -- batch of outputs

        """
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs_values,
        })

    @check_inputs_values
    def top_k(self, inputs_values, k):
        """Top k outputs.

        Arguments:
            inputs_values -- batch of inputs
            k -- top outputs count

        Return:
            top_k_values -- batch of top k outputs

        """
        return self.sess.run(tf.nn.top_k(self.outputs, k=k), feed_dict={
            self.inputs: inputs_values,
        })

    def __str__(self):
        string = 'TFNeuralNetwork object:\n'
        for attr in self.__slots__:
            if hasattr(self, attr):
                string += "%20s: %s\n" % (attr, getattr(self, attr))
        return string[:-1]
