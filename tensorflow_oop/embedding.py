import warnings
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '../')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.neural_network import *

class TFTripletset(TFDataset):

    """
    Triplet generation dataset.
    """

    __slots__ = TFDataset.__slots__ + ['batch_positives_count_', 'batch_negatives_count_']

    def initialize(self, data, labels):
        """Set data and labels."""
        assert data is not None and labels is not None, \
            'Data and labels should be passed: data = %s, labels = %s' % (data, labels)
        ndim = np.asarray(labels).ndim
        assert ndim == 1, \
            'Labels should be 1D dimension: labels.ndim = %s' % ndim
        super(TFTripletset, self).initialize(data, labels)
    
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set."""
        train_set, val_set, test_set = super(TFTripletset, self).split(train_size, val_size, test_size, shuffle)
        if train_set is not None:
            train_set = TFTripletset(data=train_set.data_, labels=train_set.labels_.flatten())
            train_set.set_batch_size(self.batch_size_, self.batch_positives_count_)
        if val_set is not None:
            val_set = TFTripletset(data=val_set.data_, labels=val_set.labels_.flatten())
            val_set.set_batch_size(self.batch_size_, self.batch_positives_count_)
        if test_set is not None:
            test_set = TFTripletset(data=test_set.data_, labels=test_set.labels_.flatten())
            test_set.set_batch_size(self.batch_size_, self.batch_positives_count_)
        return train_set, val_set, test_set

    @check_initialization
    def set_batch_size(self, batch_size, batch_positives_count):
        """Set batch size and positives count per batch."""
        assert batch_positives_count > 0, \
            'Positives count in batch should be greater than zero: batch_positives_count = %s' % batch_positives_count
        assert batch_positives_count < batch_size, \
            'Positives count in batch should be less than batch size: batch_positives_count = %s, batch_size = %s' % (batch_positives_count, batch_size)
        super(TFTripletset, self).set_batch_size(batch_size)
        self.batch_positives_count_ = int(batch_positives_count)
        self.batch_negatives_count_ = self.batch_size_ - self.batch_positives_count_

    @check_initialization
    def next_batch(self):
        """Get next batch."""
        labels = self.labels_.flatten()
        labels_counts = np.bincount(labels)
        positive_keys = np.where(labels_counts >= self.batch_positives_count_)[0]
        positive_key = positive_keys[np.random.randint(0, len(positive_keys))]

        # Take positive samples
        positives = self.data_[labels == positive_key]
        positives = positives[np.random.choice(np.arange(labels_counts[positive_key]), self.batch_positives_count_, replace=False), :]

        # Take negative samples
        negatives = self.data_[labels != positive_key]
        negatives = negatives[np.random.choice(np.arange(negatives.shape[0]), self.batch_negatives_count_, replace=False), :]

        batch_data = np.vstack([positives, negatives])
        batch_labels = np.append(np.zeros(len(positives)), np.ones(len(negatives)))
        return TFBatch(data_=batch_data, labels_=batch_labels)

class TFEmbedding(TFNeuralNetwork):

    """
    Embedding model with Triplet loss function.
    """

    @staticmethod
    def squared_distance(first_points, second_points):
        """Pairwise squared distances between 2 sets of points."""
        diff = tf.squared_difference(tf.expand_dims(first_points, 1), second_points)
        return tf.reduce_sum(diff, axis=2)

    def __init__(self, log_dir, inputs_shape, outputs_shape, inputs_type=tf.float32, outputs_type=tf.float32, reset_default_graph=True, metric_functions={}, **kwargs):
        if len(metric_functions) == 0:
            def max_accuracy(outputs, labels_placeholder):
                """Pairwise binary classification accuracy."""
                # Calculate distances
                embedding_pos, embedding_neg = tf.dynamic_partition(outputs, partitions=tf.reshape(labels_placeholder, [-1]), num_partitions=2)
                pos_dist = TFEmbedding.squared_distance(embedding_pos, embedding_pos)
                neg_dist = TFEmbedding.squared_distance(embedding_pos, embedding_neg)
                tf.summary.histogram('pos_dist', pos_dist)
                tf.summary.histogram('neg_dist', neg_dist)

                def triplet_accuracy(pos_dist, neg_dist):
                    """Triplet accuracy function for binary classification to positives and negatives."""
                    def accuracy(threshold):
                        correct_count = tf.count_nonzero(pos_dist < threshold) + tf.count_nonzero(neg_dist >= threshold)
                        total_count = tf.shape(pos_dist)[0] * tf.shape(pos_dist)[1] + tf.shape(neg_dist)[0] * tf.shape(neg_dist)[1]
                        return tf.cast(correct_count, dtype=tf.float32) / tf.cast(total_count, dtype=tf.float32)
                    return accuracy

                # Get all possible threshold values
                total_dist = tf.reshape(tf.concat([pos_dist, neg_dist], 1), [-1])
                thresholds = tf.unique(total_dist)[0]

                # Calculate accuracy
                accuracies = tf.map_fn(triplet_accuracy(pos_dist, neg_dist), thresholds)
                return tf.reduce_max(accuracies)

            metric_functions['max_accuracy'] = max_accuracy

        # Reset default graph.
        if reset_default_graph:
            tf.reset_default_graph()

        self.labels_placeholder_ = tf.placeholder(tf.int32, shape=[None], name='input_labels')
        super(TFEmbedding, self).__init__(log_dir, inputs_shape, outputs_shape, inputs_type=inputs_type, outputs_type=outputs_type, reset_default_graph=False, metric_functions=metric_functions, **kwargs)

    def loss_function(self, outputs, labels_placeholder, **kwargs):
        """Compute the triplet loss by mini-batch of triplet embeddings."""
        assert 'margin' in kwargs, \
            'Argument \'margin\' should be passed: kwargs = %s' % kwargs
        assert 'exclude_hard' in kwargs, \
            'Argument \'exclude_hard\' should be passed: kwargs = %s' % kwargs
        margin = kwargs['margin']
        exclude_hard = kwargs['exclude_hard']

        def triplet_loss(margin, exclude_hard):
            """Triplet loss function for a given anchor."""
            def loss(pos_neg_dist):
                pos_dist, neg_dist = pos_neg_dist
                raw_loss = tf.expand_dims(pos_dist, -1) - neg_dist + margin
                mask = raw_loss > 0
                if exclude_hard:
                    mask = tf.logical_and(mask, raw_loss < margin)
                valid_loss = raw_loss * tf.cast(mask, dtype=tf.float32)
                return valid_loss
            return loss

        # Calculate distances
        embedding_pos, embedding_neg = tf.dynamic_partition(outputs, partitions=self.labels_placeholder_, num_partitions=2)
        pos_dist = TFEmbedding.squared_distance(embedding_pos, embedding_pos)
        neg_dist = TFEmbedding.squared_distance(embedding_pos, embedding_neg)

        # Calculate losses
        losses = tf.map_fn(triplet_loss(margin, exclude_hard), (pos_dist, neg_dist), dtype=tf.float32)
        return tf.reduce_mean(losses)

    def fit(self, train_set, iteration_count,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            epoch_count=None,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000):
        """Train model."""
        
        assert isinstance(train_set, TFTripletset), \
            'Training set should be object of TFTripletset type: type(train_set) = %s' % type(train_set)
        if val_set is not None:
            assert isinstance(val_set, TFTripletset), \
                'Validation set should be object of TFTripletset type: type(val_set) = %s' % type(val_set)

        super(TFEmbedding, self).fit(train_set=train_set, iteration_count=iteration_count,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    epoch_count=epoch_count,
                    val_set=val_set,
                    summarizing_period=summarizing_period,
                    logging_period=logging_period,
                    checkpoint_period=checkpoint_period,
                    evaluation_period=evaluation_period)

    def evaluate(self, dataset):
        """Evaluate model."""
        if isinstance(dataset, TFDataset):
            warnings.warn('Evaluation function is not implemented for TFDataset!', Warning)
            return {}
        else:
            return super(TFEmbedding, self).evaluate(dataset)

    @check_inputs_values
    def visualize(self, inputs_values, var_name, labels=None):
        """Visualize embeddings in TensorBoard."""
        if labels is not None:
            assert len(inputs_values) == len(labels), \
                'Inputs values and labels should be the same lengths: len(inputs_values) = %s, len(labels) = %s' % (len(inputs_values), len(labels))

        # Get visualization embeddings
        vis_embeddings = self.forward(inputs_values)
        if labels is not None:
            vis_labels = labels.flatten()
        vis_name = tf.get_default_graph().unique_name(var_name, mark_as_used=False)
        
        # Input set for Embedded TensorBoard visualization
        vis_var = tf.Variable(tf.stack(vis_embeddings, axis=0), trainable=False, name=vis_name)
        self.sess_.run(tf.variables_initializer([vis_var]))

        # Add embedding tensorboard visualization
        embed = self.projector_config_.embeddings.add()
        embed.tensor_name = vis_name + ':0'
        if labels is not None:
            embed.metadata_path = os.path.join(self.log_dir_, embed.tensor_name + '_metadata.tsv')
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(self.summary_writer_, self.projector_config_)

        # Checkpoint configuration
        checkpoint_name = 'vis-checkpoint'
        checkpoint_file = os.path.join(self.log_dir_, checkpoint_name)

        # Save checkpoint
        self.save(checkpoint_file)

        # Write labels info
        if labels is not None:
            with open(embed.metadata_path, 'w') as f:
                is_first = True
                for label in vis_labels:
                    if is_first:
                        f.write(str(label))
                        is_first = False
                    else:
                        f.write('\n' + str(label))

        # Print status info
        print('For watching in TensorBoard run command:\ntensorboard --logdir "%s"' % self.log_dir_)
