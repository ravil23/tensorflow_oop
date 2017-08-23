from neural_network import *

class TFRegressor(TFNeuralNetwork):

    """
    Regression model with Mean squared loss function.
    """

    def loss_function(self, outputs, labels_placeholder):
        """Mean squared error."""
        return tf.losses.mean_squared_error(labels_placeholder, outputs)
