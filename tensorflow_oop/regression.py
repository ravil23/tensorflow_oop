"""
Regression base models.
"""

from tensorflow_oop.neural_network import *


class TFRegressor(TFNeuralNetwork):

    """
    Regression model with Mean squared loss function.

    Attributes:
        ...         Parrent class atributes.

    """

    def loss_function(self, targets, outputs):
        """Mean squared error.

        Arguments:
            targets     Tensor of batch with targets.
            outputs     Tensor of batch with outputs.

        Return:
            loss        Mean squared error operation.

        """
        return tf.losses.mean_squared_error(targets, outputs)
