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
    Classification model with Cross entropy loss function.
    """

    __slots__ = TFNeuralNetwork.__slots__ + ['probabilities']

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
        super(TFClassifier, self).initialize(inputs_shape=inputs_shape,
                                             targets_shape=targets_shape,
                                             outputs_shape=outputs_shape,
                                             inputs_type=inputs_type,
                                             targets_type=targets_type,
                                             outputs_type=outputs_type,
                                             reset=reset,
                                             **kwargs)

        # Add probability operation
        self.probabilities = tf.nn.softmax(self.outputs)

        # Add accuracy metric
        def accuracy_function(targets, outputs):
            correct_prediction = tf.equal(tf.argmax(targets, 1),
                                          tf.argmax(outputs, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.add_metric('accuracy', accuracy_function)

    def loss_function(self, targets, outputs, **kwargs):
        """Cross entropy.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of keyword arguments

        Return:
            loss -- cross entropy error operation

        """
        return tf.losses.softmax_cross_entropy(labels_placeholder, outputs) 

    @check_inputs_values
    def probabilities(self, inputs_values):
        """Get probabilities.

        Arguments:
            inputs_values -- batch of inputs

        Return:
            probability_values -- batch of probabilities

        """
        return self.sess.run(self.probabilities, feed_dict={
            self.inputs: inputs_values,
        })

    @check_inputs_values
    def predict(self, inputs_values):
        """Get predictions corresponded to maximum probabilities.

        Arguments:
            inputs_values -- batch of inputs

        Return:
            prediction_values -- batch of predictions

        """
        return self.sess.run(tf.argmax(self.probabilities, 1), feed_dict={
            self.inputs: inputs_values,
        })
