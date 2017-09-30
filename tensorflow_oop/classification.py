"""
Classification base models.
"""

from tensorflow_oop.neural_network import *


class TFClassifier(TFNeuralNetwork):

    """
    Classification model with Softmax and Cross entropy loss function.

    Attributes:
        ...                Parrent class atributes.
        classes_count      Classification classes count.
        one_hot_targets    Tensor with one hot representation of targets.
        softmax            Softmax layer.
        predictions        Classes indices corresponded to maximum of softmax.
        top_k_placeholder  Placeholder for parameter k in top prediction metrics.
        top_k_softmax      Top k values and indices of softmax.

    """

    __slots__ = TFNeuralNetwork.__slots__ + ['classes_count', 'one_hot_targets',
                                             'softmax', 'predictions',
                                             'top_k_placeholder', 'top_k_softmax']

    def initialize(self,
                   classes_count,
                   inputs_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   outputs_type=tf.float32,
                   print_self=True,
                   k_values=None,
                   **kwargs):
        """Initialize model.

        Arguments:
            classes_count      Classification classes count.
            inputs_shape       Shape of inputs layer without batch dimension.
            outputs_shape      Shape of outputs layer without batch dimension.
            inputs_type        Type of inputs layer.
            outputs_type       Type of outputs layer.
            print_self         Indicator of printing model after initialization.
            k_values           Container of k values for top prediction metrics.
            kwargs             Dict object of options.

        """
        super(TFClassifier, self).initialize(inputs_shape=inputs_shape,
                                             targets_shape=[],
                                             outputs_shape=outputs_shape,
                                             inputs_type=inputs_type,
                                             targets_type=tf.int32,
                                             outputs_type=outputs_type,
                                             print_self=False,
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

        if print_self:
            print('%s\n' % self)

    def loss_function(self, targets, outputs):
        """Cross entropy for only one correct answer.

        Arguments:
            targets     Tensor of batch with targets.
            outputs     Tensor of batch with outputs.

        Return:
            loss        Cross entropy error operation.

        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=outputs)
        return tf.reduce_mean(cross_entropy)

    @check_initialization
    @check_inputs_values
    def predict(self, inputs_values):
        """Get predictions corresponded to maximum probabilities.

        Arguments:
            inputs_values      Batch of inputs values.

        Return:
            probabilities      Batch of all probabilities.
            best_indices       Batch of best prediction indices.

        """
        return self.sess.run([self.probabilities, self.predictions], feed_dict={
            self.inputs: inputs_values,
        })

    @check_initialization
    @check_inputs_values
    def predict_top_k(self, inputs_values, k):
        """Get predictions corresponded to top k probabilities.

        Arguments:
            inputs_values      Batch of inputs values.
            k                  Count of top predictions.

        Return:
            k_probabilities    Batch of top k probabilities.
            k_indices          Batch of top k prediction indices.

        """
        return self.sess.run(self.top_k_softmax, feed_dict={
            self.inputs: inputs_values,
            self.top_k_placeholder: k
        })

    def _add_basic_classification_metrics(self, k_values=None):
        """Add basic classification metrics as top accuracy.

        Arguments:
            k_values           Container of k values for top prediction metrics,
                               if not passed used only best prediction as top 1.

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
