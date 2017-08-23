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

    def loss_function(self, outputs, labels_placeholder):
        """Mean squared error."""
        return tf.losses.mean_squared_error(labels_placeholder, outputs)
