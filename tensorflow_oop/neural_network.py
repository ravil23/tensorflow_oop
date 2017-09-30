"""
Neural network class -- parent for all models.
"""

import tensorflow as tf
import numpy as np
import os
import time
import sys
import warnings
from tensorflow.contrib.tensorboard.plugins import projector

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.dataset import *
from tensorflow_oop.decorators import *

# Set logging level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TFNeuralNetwork(object):

    """
    Basic neural network model.
    """

    __slots__ = ['init', 'restored', 'log_dir',
                 'inputs_shape', 'targets_shape', 'outputs_shape',
                 'inputs', 'targets', 'outputs',
                 'loss', 'dropout', 'global_step',
                 'sess',
                 'options', 'metrics', '_update_ops', '_summaries',
                 '_summary_writer', '_projector_config', '_saver',
                 '_best_val_checkpoint', '_best_val_result',
                 '_best_val_iteration', '_best_val_key',
                 '_fit_checkpoint', '_vis_checkpoint',
                 '_iteration', '_iter_count',
                 '_epoch', '_epoch_count',
                 '_local_variables_initializer']

    def __init__(self, log_dir, clear, reset_graph=True):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.log_dir = log_dir
        self.init = False
        self.restored = False
        self.options = {}
        self.metrics = {'batch_train': {},
                        'batch_validation': {},
                        'log_train': {},
                        'eval_train': {},
                        'eval_validation': {},
                        'eval_test': {}}
        self._update_ops = {'batch_train': {},
                        'batch_validation': {},
                        'log_train': {},
                        'eval_train': {},
                        'eval_validation': {},
                        'eval_test': {}}
        self._summaries = {'batch_train': [],
                        'batch_validation': [],
                        'log_train': [],
                        'eval_train': [],
                        'eval_validation': [],
                        'eval_test': []}

        # Checkpoint paths
        self._fit_checkpoint = os.path.join(self.log_dir, 'fit-checkpoint')
        self._vis_checkpoint = os.path.join(self.log_dir, 'vis-checkpoint')
        self._best_val_checkpoint = os.path.join(self.log_dir, 'best-val-checkpoint')

        # Reset default graph if necessary
        if reset_graph:
            tf.reset_default_graph()

        # Reset if necessary
        if clear:
            # Clean TensorBoard logging directory
            if tf.gfile.Exists(self.log_dir):
                tf.gfile.DeleteRecursively(self.log_dir)

    def initialize(self,
                   inputs_shape,
                   targets_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   targets_type=tf.float32,
                   outputs_type=tf.float32,
                   print_self=True,
                   **kwargs):
        """Initialize model.

        Arguments:
            inputs_shape -- shape of inputs layer
            targets_shape -- shape of targets layer
            outputs_shape -- shape of outputs layer
            inputs_type -- type of inputs layer
            targets_type -- type of targets layer
            outputs_type -- type of outputs layer
            print_self -- indicator of printing model after initialization
            kwargs -- dictionary of keyword arguments

        """
        print('Start initializing model...')

        # Create TensorBoard logging directory
        if not tf.gfile.Exists(self.log_dir):
            tf.gfile.MakeDirs(self.log_dir)

        # Create a session for running Ops on the Graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Options
        for key in kwargs:
            self.add_option_to_graph(key, kwargs[key])

        # Input, Target and Output layer shapes
        self.inputs_shape = list(inputs_shape)
        self.targets_shape = list(targets_shape)
        self.outputs_shape = list(outputs_shape)

        # Generate placeholders for the inputs and targets
        self.inputs = tf.placeholder(inputs_type,
                                     shape=[None] + self.inputs_shape,
                                     name='inputs')
        self.targets = tf.placeholder(targets_type,
                                      shape=[None] + self.targets_shape,
                                      name='targets')

        # Build a Graph that computes predictions from the inference model
        outputs = self.inference(self.inputs)
        self.outputs = tf.identity(outputs, name='outputs')

        # Loss function
        loss = self.loss_function(self.targets, self.outputs)
        self.loss = tf.identity(loss, name='loss')

        # Dropout tensor
        self.dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')

        # Global step tensor
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add loss metric
        self.add_metric(self.loss, collections=['batch_train',
                                                'batch_validation',
                                                'log_train'])

        # Instantiate a SummaryWriter to output summaries and the Graph
        self._summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # Projector config object
        self._projector_config = projector.ProjectorConfig()

        # Run the Op to initialize the variables
        self.initialize_variables(init_global=True, init_local=True)

        # Enable initialization flag
        self.init = True

        print('Finish initializing model.')
        if print_self:
            print('%s\n' % self)

    def add_option_to_graph(self, name, value):
        """Add option to graph.

        Arguments:
            name -- name of tensor
            value -- data value for tensor

        Return:
            option -- tensorflow variable

        """
        option = None
        try:
            option = tf.Variable(value, name='options/' + name, trainable=False)
            self.sess.run(tf.variables_initializer([option]))
        except:
            warnings.warn('''Option '%s' can't be saved to graph as variable.''' % name)
        self.options[name] = value
        return option

    def initialize_variables(self, init_global, init_local):
        """Initialize uninitialized global, all local variables and create new saver.

        Arguments:
            init_global -- boolean indicator of initialization global variables
            init_local -- boolean indicator of initialization local variables

        """
        if init_global:
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    self.sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            if len(uninitialized_vars) > 0:
                init_new_vars_op = tf.variables_initializer(uninitialized_vars)
                self.sess.run(init_new_vars_op)
                self._saver = tf.train.Saver(max_to_keep=1000)
        if init_local:
            self._local_variables_initializer = tf.local_variables_initializer()
            self.sess.run(self._local_variables_initializer)            

    @check_add_metric_arguments
    def add_metric(self, metric, collections, key=None):
        """Add logging and summarizing metric.

        Arguments:
            metric -- tensorflow operation
            collections -- list of strings from ['batch_train',
                                                 'batch_validation',
                                                 'log_train',
                                                 'eval_train',
                                                 'eval_validation',
                                                 'eval_test']
            key -- string key

        """

        # Get metric tensor and operation
        if isinstance(metric, tf.Tensor):
            update_op = None
        else:
            metric, update_op = metric

        # Parse metric key
        if key is None:
            key = self._basename_tensor(metric)

        # Auto detect summary type
        metric_rank = tf.rank(metric).eval(session=self.sess)
        if metric_rank == 0:
            summary_type = tf.summary.scalar
        else:
            summary_type = tf.summary.histogram

        # Add metric to passed collections
        for collection in collections:
            if 'eval' in collection or update_op is None:
                # Add summary to graph
                summary_type(collection + '/' + key,
                             metric,
                             collections=[collection])

                # Add metric to dictionary
                self.metrics[collection][key] = metric
            else:
                # Add summary to graph
                summary_type(collection + '/' + key,
                             update_op,
                             collections=[collection])

                # Add metric to dictionary
                self.metrics[collection][key] = update_op

            # Update summaries dictionary
            self._summaries[collection] = tf.summary.merge_all(collection)

            # Add tensors to graph collection
            self.sess.graph.add_to_collection('metric_' + collection, metric)
            if update_op is not None:
                self.sess.graph.add_to_collection('update_op_' + collection, update_op)

            # Add metric to dictionary
            if update_op is not None:
                self._update_ops[collection][key] = update_op

        # Initialize new variables
        self.initialize_variables(init_global=True, init_local=True)

    def _get_train_op(self, optimizer, learning_rate, max_gradient_norm):
        """Get training operation.

        Arguments:
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step
            max_gradient_norm -- maximal gradient norm for clipping

        Return:
            trin_op -- training operation

        """
        optimizer_op = optimizer(learning_rate)

        # Calculate gradients
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars)

        # Add tvars metric
        flatten_tvars = []
        for tvar in tvars:
            flatten_tvars.append(tf.reshape(tvar, [-1,]))
        concat_tvars = tf.concat(flatten_tvars, 0, name='all_tvars')
        self.add_metric(concat_tvars, collections=['batch_train'])

        # Add gradients metric
        flatten_gradients = []
        for gradient in gradients:
            flatten_gradients.append(tf.reshape(gradient, [-1,]))
        concat_gradients = tf.concat(flatten_gradients, 0,
                                     name='all_gradients')
        self.add_metric(concat_gradients, collections=['batch_train'])

        # Gradient clipping if necessary
        if max_gradient_norm is not None:
            # Calculate clipping gradients
            clip_gradients, gradient_norm = tf.clip_by_global_norm(gradients, 
                                                                   max_gradient_norm)
            
            # Add clipping gradients metric
            flatten_clip_gradients = []
            for clip_gradient in clip_gradients:
                flatten_clip_gradients.append(tf.reshape(clip_gradient, [-1,]))
            concat_clip_gradients = tf.concat(flatten_clip_gradients, 0)
            self.add_metric(concat_clip_gradients, collections=['batch_train'],
                            key='all_clip_gradients')
            self.add_metric(gradient_norm, collections=['batch_train'],
                            key='gradient_norm')

            # Add to the Graph the Ops that apply gradients
            train_op = optimizer_op.apply_gradients(zip(clip_gradients, tvars),
                                                    global_step=self.global_step,
                                                    name='train_op')
        else:
            # Add to the Graph the Ops that minimize loss
            train_op = optimizer_op.minimize(self.loss,
                                             global_step=self.global_step,
                                             name='train_op')

        # Run the Op to initialize the variables
        self.initialize_variables(init_global=True, init_local=True)

        return train_op

    @check_initialization
    @check_fit_arguments
    def fit(self,
            train_set,
            val_set=None,
            epoch_count=None,
            iter_count=None,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            max_gradient_norm=None,
            summarizing_period=100,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000,
            best_val_key=None):
        """Train model.

        Arguments:
            train_set -- dataset for training
            val_set -- dataset for validation
            epoch_count -- training epochs count
            iter_count -- training iterations count
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step numeric or tensor
            max_gradient_norm -- maximal gradient norm for clipping
            summarizing_period -- iterations count between summarizing
            logging_period -- iterations count between logging to stdout
            checkpoint_period -- iterations count between saving checkpoint
            evaluation_period -- iterations count between evaluation
            best_val_key -- metric key for saving best validation checkpoint

        """
        print('Start training...')
        start_fit_time = time.time()

        # Update actual iter and epoch count
        self._get_actual_iter_epoch_count(iter_count, epoch_count,
                                          train_set.size, train_set.batch_size)

        # Global iter step and epoch number
        self._iteration = self.global_step.eval(session=self.sess)
        self._epoch = self._iteration * train_set.batch_size // train_set.size
        assert self._iter_count >= self._iteration, \
            '''Iteration count should be greater than init iteration:
            self._iter_count = %s, self._iteration = %s''' \
            % (self._iter_count, self._iteration)
        if self._iter_count == self._iteration:
            print('Init iteration is equal to iteration count.\n')
            return

        # Get training operation
        train_op = self._get_train_op(optimizer, learning_rate, max_gradient_norm)

        # Print fitting options
        self._print_fitting_options(train_set.batch_size,
                                    val_set.batch_size if val_set else None,
                                    self._epoch_count,
                                    self._iter_count,
                                    optimizer,
                                    learning_rate,
                                    max_gradient_norm,
                                    summarizing_period,
                                    logging_period,
                                    checkpoint_period,
                                    evaluation_period,
                                    best_val_key)

        # Calculate initial result on validation set
        if val_set is not None and best_val_key is not None:
            print('Initial evaluation...')
            self._best_val_key = best_val_key
            self.evaluate_and_log(val_set, 'eval_validation')

        # Initial logging period time
        self._last_log_time = time.time()

        # Loop over all batches
        while self._iteration < self._iter_count:
            # Get current iteration and epoch
            self._iteration += 1
            self._epoch = self._iteration * train_set.batch_size // train_set.size

            # Calculate current iteration options
            last_iteration_flag = self._iteration == self._iter_count
            summarizing_flag = self._iteration % summarizing_period == 0 or last_iteration_flag
            logging_flag     = self._iteration % logging_period     == 0 or last_iteration_flag
            checkpoint_flag  = self._iteration % checkpoint_period  == 0 or last_iteration_flag
            evaluation_flag  = self._iteration % evaluation_period  == 0 or last_iteration_flag

            # One training iteration
            self._training_step(train_set, train_op, summarizing_flag, logging_flag)

            # One validation iteration
            if val_set is not None:
                self._validation_step(val_set, summarizing_flag)

            # Save a checkpoint the model periodically
            if checkpoint_flag:
                print('Saving checkpoint periodically...')
                self.save(self._fit_checkpoint, global_step=self._iteration)

            # Evaluate the model periodically
            if evaluation_flag:
                print('Evaluation...')
                self.evaluate_and_log(train_set, 'eval_train')

                # Eval on validation set if necessary
                if val_set is not None:
                    self.evaluate_and_log(val_set, 'eval_validation')

        self._summary_writer.flush()
        total_time = time.time() - start_fit_time
        print('Finish training iteration (total time %.3f sec).\n' % total_time)

        if val_set is not None and best_val_key is not None:
            print('Report by best result on validation set:')
            print('%20s : %s' % ('best_val_key', self._best_val_key))
            print('%20s : %s' % ('best_val_result', self._best_val_result))
            print('%20s : %s' % ('best_val_iteration', self._best_val_iteration))
            print('%20s : %s\n' % ('best_val_checkpoint', self._best_val_checkpoint))

    @check_initialization
    def fill_feed_dict(self, batch):
        """Get filled feed dictionary for batch.

        Arguments:
            batch -- batch of inputs

        """
        feed_dict = {
            self.inputs: batch.data,
            self.targets: batch.labels,
        }
        return feed_dict

    @check_initialization
    @check_produce_arguments
    def produce(self, dataset, batch, output_tensors):
        """Produce model on batch and return list of output tensors values.

        Arguments:
            dataset -- TFDataset object
            batch -- TFBatch object
            output_tensors -- container of output tensors

        Return:
            output_values -- list of output tensors values

        """

        # Fill feed dict
        feed_dict = self.fill_feed_dict(batch)

        # Run one step of the model
        output_values = self.sess.run(output_tensors, feed_dict=feed_dict)

        return output_values

    @check_initialization
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

    @check_initialization
    @check_evaluate_arguments
    def evaluate(self, dataset, collection):
        """Evaluate model.

        Arguments:
            dataset -- TFDataset object
            collection -- string value from ['eval_train',
                                             'eval_validation',
                                             'eval_test']

        Return:
            metrics -- dictionary

        """

        if len(self.metrics[collection]) == 0:
            return {}

        # Reset local metric tickers
        self.sess.run(self._local_variables_initializer)

        # Update local variables by one epoch
        for batch in dataset.iterbatches(count=None):
            self.produce(dataset, batch, self._update_ops[collection])

        # Calculate metrics values
        full_batch = dataset.full_batch()
        output_tensors = [self.metrics[collection], self._summaries[collection]]
        metrics, summary_str = self.produce(dataset, full_batch, output_tensors)

        # Update the events file with evaluation summary
        self._summary_writer.add_summary(summary_str, self._iteration)

        # Save best result on validation set if necessary
        if self._best_val_key is not None and collection == 'eval_validation':
            result = metrics[self._best_val_key]
            if self._best_val_result is None or result > self._best_val_result:
                print('Saving checkpoint with best result on validation set...')
                self.save_best_on_validation(result)

        return metrics

    @check_initialization
    @check_evaluate_arguments
    def evaluate_and_log(self, dataset, collection):
        """Evaluate model.

        Arguments:
            dataset -- TFDataset object
            collection -- string value from ['eval_train',
                                             'eval_validation',
                                             'eval_test']

        Return:
            metrics -- dictionary

        """

        # Evaluate on current collection
        start_evaluation_time = time.time()
        metrics = self.evaluate(dataset, collection)
        duration = time.time() - start_evaluation_time

        if len(metrics) > 0:
            # Convert metrics to string
            keys = sorted(list(metrics.keys()))
            metrics_str = '   '.join(['%s = %.6f' % (k, metrics[k]) for k in keys])

            # Log evaluation result
            format_string = 'Evaluation on [%s]:   %s   [%.3f sec]'
            print(format_string % (collection, metrics_str, duration))
        return metrics

    @check_initialization
    def save(self, filename, global_step=None):
        """Save checkpoint.

        Arguments:
            filename -- path to saving
            global_step -- optional suffix adding to path (default None)

        """
        saved_filename = self._saver.save(self.sess, filename,
                                          global_step=global_step)
        print('Model saved to: %s' % saved_filename)

    @check_initialization
    def save_best_on_validation(self, result):
        """Save checkpoint with best result on validation set.

        Arguments:
            result -- new best validation result

        """
        self._best_val_result = result
        self._best_val_iteration = self._iteration
        self.save(self._best_val_checkpoint)

    @check_initialization
    def restore(self, filename=None):
        """Restore checkpoint only if model initialized.
        
        Arguments:
            filename -- path to checkpoint
        
        """
        if filename is None:
            filename = tf.train.latest_checkpoint(self.log_dir)
            assert filename is not None, \
                'Checkpoint path automatically not found.'

        self._saver.restore(self.sess, filename)
        self.restored = True

    @check_initialization
    def restore_best_on_validation(self):
        """Restore checkpoint with best result on validation set."""
        self.restore(self._best_val_checkpoint)
        self.restored = True

    def _get_actual_iter_epoch_count(self, iter_count, epoch_count, dataset_size, batch_size):
        """Actualize iteration and epoch count.

        Arguments:
            iter_count -- iteration count (None if unknown)
            epoch_count -- epoch count (None if unknown)
            dataset_size -- length of dataset
            batch_size -- length of batch

        """
        if epoch_count is not None:
            iter_count_by_epoch = (dataset_size * epoch_count) // batch_size
            if dataset_size % batch_size != 0:
                iter_count_by_epoch += 1
            if iter_count is not None:
                iter_count = min(iter_count, iter_count_by_epoch)
            else:
                iter_count = iter_count_by_epoch
        else:
            epoch_count = (iter_count * batch_size) // dataset_size
        self._iter_count = iter_count
        self._epoch_count = epoch_count

    def _print_fitting_options(self,
                               train_batch_size,
                               val_batch_size,
                               epoch_count,
                               iter_count,
                               optimizer,
                               learning_rate,
                               max_gradient_norm,
                               summarizing_period,
                               logging_period,
                               checkpoint_period,
                               evaluation_period,
                               best_val_key):
        """Formatted print training options."""
        print('%20s: %s' % ('train_batch_size', train_batch_size))
        print('%20s: %s' % ('val_batch_size', val_batch_size))
        print('%20s: %s' % ('epoch_count', epoch_count))
        print('%20s: %s' % ('iter_count', iter_count))
        print('%20s: %s' % ('optimizer', optimizer))
        print('%20s: %s' % ('learning_rate', learning_rate))
        print('%20s: %s' % ('max_gradient_norm', max_gradient_norm))
        print('%20s: %s' % ('summarizing_period', summarizing_period))
        print('%20s: %s' % ('logging_period', logging_period))
        print('%20s: %s' % ('checkpoint_period', checkpoint_period))
        print('%20s: %s' % ('evaluation_period', evaluation_period))
        print('%20s: %s' % ('best_val_key', best_val_key))
        buf = ''
        collections = sorted(list(self.metrics.keys()))
        for collection in collections:
            keys = list(self.metrics[collection].keys())
            buf += '%30s: %s\n' % (collection, sorted(keys))
        print('%20s:\n%s' % ('metrics', buf))

    def _basename_tensor(self, tensor):
        """Get tensor basename without scope and identification postfix.

        Arguments:
            tensor -- graph node with name attribute

        Return:
            basename -- string

        """
        name = tensor.name
        basename = str(name[name.rfind('/') + 1 : name.rfind(':')])
        if basename[-1].isdigit():
            while basename[-1].isdigit():
                basename = basename[:-1]
            basename = basename[:-1]
        return basename

    def _training_step(self, train_set, train_op, summarizing_flag, logging_flag):
        """Run one training iteration.

        Arguments:
            train_set -- TFDataset object
            train_op -- training operation
            summarizing_flag -- boolean indicator of summarizing
            logging_flag -- boolean indicator of logging

        """

        # Reset local metric tickers
        self.sess.run(self._local_variables_initializer)

        # Get next training batch
        train_batch = train_set.next_batch()

        output_tensors = [train_op, [], {}]

        # Get summarizing tensors
        if summarizing_flag:
            output_tensors[1] = self._summaries['batch_train']

        # Get logging tensors
        if logging_flag:
            output_tensors[2] = self.metrics['log_train']

        # Calculate all produced values
        _, summary_str, metrics = self.produce(train_set, train_batch, output_tensors)

        if summarizing_flag:
            # Write summaries
            self._summary_writer.add_summary(summary_str, self._iteration)

        if logging_flag:
            # Convert metrics to string
            keys = sorted(list(metrics.keys()))
            metrics_str = '   '.join(['%s = %.6f' % (k, metrics[k]) for k in keys])

            # Calculate time of last logging period
            period_time = time.time() - self._last_log_time
            self._last_log_time = time.time()

            # Log training process
            format_string = 'Iter %d / %d (epoch %d / %d):   %s   [%.3f sec]'
            print(format_string % (self._iteration,
                                   self._iter_count,
                                   self._epoch,
                                   self._epoch_count,
                                   metrics_str,
                                   period_time))

    def _validation_step(self, val_set, summarizing_flag):
        """Run one training iteration.

        Arguments:
            val_set -- TFDataset object
            summarizing_flag -- boolean indicator of summarizing

        """

        if summarizing_flag:
            # Reset local metric tickers
            self.sess.run(self._local_variables_initializer)

            # Get next validation batch
            val_batch = val_set.next_batch()

            # Produce this dataset over network
            summary_str = self.produce(val_set, val_batch, self._summaries['batch_validation'])

            self._summary_writer.add_summary(summary_str, self._iteration)

    def inference(self, inputs):
        """Model inference.

        Arguments:
            inputs -- tensor of batch with inputs

        Return:
            outputs -- tensor of outputs layer

        """
        raise Exception('Inference function should be overwritten!')
        return outputs

    def loss_function(self, targets, outputs):
        """Loss function.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs

        Return:
            loss -- tensorflow operation for minimization

        """
        raise Exception('Loss function should be overwritten!')
        return loss

    def __str__(self):
        string = 'TFNeuralNetwork object:\n'
        for attr in self.__slots__:
            if hasattr(self, attr) and attr[0] != '_':
                if attr == 'metrics':
                    buf = ''
                    collections = sorted(list(self.metrics.keys()))
                    for collection in collections:
                        keys = list(self.metrics[collection].keys())
                        buf += '%30s: %s\n' % (collection, sorted(keys))
                    string += '%20s:\n%s' % (attr, buf)
                elif attr == 'options':
                    buf = ''
                    keys = sorted(list(self.options.keys()))
                    for key in keys:
                        buf += '%30s: %s\n' % (key, self.options[key])
                    string += '%20s:\n%s' % (attr, buf)
                else:
                    string += '%20s: %s\n' % (attr, getattr(self, attr))
        return string[:-1]
