"""
Classification base models.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.neural_network import *

class TFClassifier(TFNeuralNetwork):

    """
    Classification model with Softmax and Cross entropy loss function.
    """

    __slots__ = TFNeuralNetwork.__slots__ + ['classes_count', 'one_hot_targets',
                                             'softmax', 'predictions',
                                             'top_k_placeholder', 'top_k_softmax']

    def load(self, model_checkpoint_path=None, k_values=None):
        """Load checkpoint.

        Arguments:
            model_checkpoint_path -- checkpoint path, search last if not passed
            k_values -- container of k values for top prediction metrics

        """
        super(TFClassifier, self).load(model_checkpoint_path=model_checkpoint_path)

        # Get probability operation
        self.classes_count = self.sess.graph.get_tensor_by_name('classes_count:0')
        self.one_hot_targets = self.sess.graph.get_tensor_by_name('one_hot_targets:0')
        self.softmax = self.sess.graph.get_tensor_by_name('softmax:0')
        self.predictions = self.sess.graph.get_tensor_by_name('predictions:0')
        self.top_k_placeholder = self.sess.graph.get_tensor_by_name('top_k_placeholder:0')
        self.top_k_softmax = self.sess.graph.get_tensor_by_name('top_k_softmax:0')

        # Add basic classification metrics
        self._add_basic_classification_metrics(k_values)

    def initialize(self,
                   classes_count,
                   inputs_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   outputs_type=tf.float32,
                   k_values=None,
                   **kwargs):
        """Initialize model.

        Arguments:
            classes_count -- maximum classes count
            inputs_shape -- shape of inputs layer
            outputs_shape -- shape of outputs layer
            inputs_type -- type of inputs layer
            outputs_type -- type of outputs layer
            k_values -- container of k values for top prediction metrics
            kwargs -- dictionary of keyword arguments

        """
        super(TFClassifier, self).initialize(inputs_shape=inputs_shape,
                                             targets_shape=[],
                                             outputs_shape=outputs_shape,
                                             inputs_type=inputs_type,
                                             targets_type=tf.int32,
                                             outputs_type=outputs_type,
                                             **kwargs)
        # Save classes count
        self.classes_count = tf.constant(classes_count, name='classes_count')

        # One hot encoding of targets
        self.one_hot_targets = tf.one_hot(self.targets, self.classes_count, name='one_hot_targets')

        # Add probability operation
        self.softmax = tf.nn.softmax(self.outputs, name='softmax')

        # Add predictions operation
        self.predictions = tf.argmax(self.outputs, 1, name='predictions')

        # Top K predictions
        self.top_k_placeholder = tf.placeholder(tf.int32, [], name='top_k_placeholder')
        self.top_k_softmax = tf.nn.top_k(self.softmax,
                                         k=self.top_k_placeholder,
                                         name='top_k_softmax')

        # Add basic classification metrics
        self._add_basic_classification_metrics(k_values)

    def loss_function(self, targets, outputs, **kwargs):
        """Cross entropy for only one correct answer.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of options

        Return:
            loss -- cross entropy error operation

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)
        return tf.reduce_mean(cross_entropy)

    @check_initialization
    @check_inputs_values
    def predict(self, inputs_values):
        """Get predictions corresponded to maximum probabilities.

        Arguments:
            inputs_values -- batch of inputs

        Return:
            probabilities -- batch of all probabilities
            indices -- batch of best predictions

        """
        return self.sess.run([self.probabilities, self.predictions], feed_dict={
            self.inputs: inputs_values,
        })

    @check_initialization
    @check_inputs_values
    def predict_top_k(self, inputs_values, k):
        """Get predictions corresponded to top k probabilities.

        Arguments:
            inputs_values -- batch of inputs
            k -- count of top predictions

        Return:
            k_probabilities -- batch of top k probabilities
            k_indices -- batch of top k predictions

        """
        return self.sess.run(self.top_k_softmax, feed_dict={
            self.inputs: inputs_values,
            self.top_k_placeholder: k
        })

    def _add_basic_classification_metrics(self, k_values=None):
        """Add basic accuracy metrics.

        Arguments:
            k_values -- container of k values for top prediction metrics, if not passed used only best prediction as top1

        """
        if k_values is not None:
            # Add top K metrics
            for k in set(k_values):
                in_top_k = tf.nn.in_top_k(self.softmax, self.targets, k=k, name='in_top_%s' % k)

                # Calculate top K accuracy
                top_k_accuracy = tf.metrics.mean(in_top_k)

                # Add top K accuracy metric
                self.add_metric(top_k_accuracy,
                                collections=['batch_train', 'batch_validation',
                                             'log_train',
                                             'eval_train', 'eval_validation', 'eval_test'],
                                key='top_%s_accuracy' % k)
        else:
            # Calculate top K accuracy
            accuracy = tf.metrics.accuracy(self.targets, self.predictions)

            # Add top K accuracy metric
            self.add_metric(accuracy,
                            collections=['batch_train', 'batch_validation',
                                         'log_train',
                                         'eval_train', 'eval_validation', 'eval_test'],
                            key='accuracy')