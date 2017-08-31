import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.neural_network import *

class TFRegressor(TFNeuralNetwork):

    """
    Regression model with Mean squared loss function.
    """

    def loss_function(self, targets, outputs, **kwargs):
        """Mean squared error.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of keyword arguments

        Return:
            loss -- mean squared error operation

        """
        return tf.losses.mean_squared_error(targets, outputs)
