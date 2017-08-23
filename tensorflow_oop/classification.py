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

    __slots__ = TFNeuralNetwork.__slots__ + ['probabilities_']

    def __init__(self, log_dir, inputs_shape, outputs_shape, inputs_type=tf.float32, outputs_type=tf.float32, reset_default_graph=True, metric_functions={}, **kwargs):
        if len(metric_functions) == 0:
            def accuracy(outputs, labels_placeholder):
                correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels_placeholder, 1))
                return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
            metric_functions['accuracy'] = accuracy

        super(TFClassifier, self).__init__(log_dir, inputs_shape, outputs_shape, inputs_type=inputs_type, outputs_type=outputs_type, reset_default_graph=reset_default_graph, metric_functions=metric_functions, **kwargs)
        self.probabilities_ = tf.nn.softmax(self.outputs_)
        
    def loss_function(self, outputs, labels_placeholder, **kwargs):
        """Cross entropy."""
        return tf.losses.softmax_cross_entropy(labels_placeholder, outputs) 

    @check_inputs_values
    def probabilities(self, inputs_values):
        """Get probabilites."""
        return self.sess_.run(self.probabilities_, feed_dict={
            self.data_placeholder_: inputs_values,
        })

    @check_inputs_values
    def classify(self, inputs_values):
        """Best prediction."""
        return self.sess_.run(tf.argmax(self.probabilities_, 1), feed_dict={
            self.data_placeholder_: inputs_values,
        })
