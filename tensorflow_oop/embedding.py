"""
Embedding base models.
"""

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

    __slots__ = TFDataset.__slots__ + ['batch_positives_count',
                                       'batch_negatives_count']

    def initialize(self, data, labels):
        """Set data and labels."""
        assert data is not None and labels is not None, \
            '''Data and labels should be passed:
            data = %s, labels = %s''' % (data, labels)

        ndim = np.asarray(labels).ndim
        assert ndim == 1, \
            '''Labels should be 1D dimension: labels.ndim = %s''' % ndim

        super(TFTripletset, self).initialize(data=data, labels=labels)

    @check_initialization
    def split(self, train_size, val_size, test_size, shuffle):
        """Split dataset to train, validation and test set."""
        train_set, val_set, test_set = super(TFTripletset, self).split(
            train_size,
            val_size,
            test_size,
            shuffle)
        if train_set is not None:
            train_set = TFTripletset(data=train_set.data,
                                     labels=train_set.labels.flatten())
            train_set.set_batch_size(self.batch_size, self.batch_positives_count)
        if val_set is not None:
            val_set = TFTripletset(data=val_set.data,
                                   labels=val_set.labels.flatten())
            val_set.set_batch_size(self.batch_size, self.batch_positives_count)
        if test_set is not None:
            test_set = TFTripletset(data=test_set.data,
                                    labels=test_set.labels.flatten())
            test_set.set_batch_size(self.batch_size, self.batch_positives_count)
        return train_set, val_set, test_set

    @check_initialization
    def set_batch_size(self, batch_size, batch_positives_count):
        """Set batch size and positives count per batch."""
        assert batch_positives_count > 0, \
            '''Positives count in batch should be greater than zero:
            batch_positives_count = %s''' % batch_positives_count

        assert batch_positives_count < batch_size, \
            '''Positives count in batch should be less than batch size:
            batch_positives_count = %s, batch_size = %s''' \
            % (batch_positives_count, batch_size)

        super(TFTripletset, self).set_batch_size(batch_size)
        self.batch_positives_count = int(batch_positives_count)
        self.batch_negatives_count = self.batch_size - self.batch_positives_count

    @check_initialization
    def next_batch(self):
        """Get next batch."""
        labels = self.labels.flatten()
        labels_counts = np.bincount(labels)
        positive_keys = np.where(labels_counts >= self.batch_positives_count)[0]
        rand_pos_key = positive_keys[np.random.randint(0, len(positive_keys))]

        def random_sample(data, count):
            indexes = np.arange(data.shape[0])
            rand_indexes = np.random.choice(indexes, count, replace=False)
            return data[rand_indexes]

        # Take positive samples
        positives = random_sample(self.data[labels == rand_pos_key],
                                  self.batch_positives_count)

        # Take negative samples
        negatives = random_sample(self.data[labels != rand_pos_key],
                                  self.batch_negatives_count)

        # Create batch
        batch_data = np.vstack([positives, negatives])
        batch_labels = np.append(np.zeros(len(positives)),
                                 np.ones(len(negatives)))
        return TFBatch(data=batch_data, labels=batch_labels)

class TFEmbedding(TFNeuralNetwork):

    """
    Embedding model with Triplet loss function.
    """

    @staticmethod
    def squared_distance(first_points, second_points):
        """Pairwise squared distances between 2 sets of points."""
        diff = tf.squared_difference(tf.expand_dims(first_points, 1),
                                     second_points)
        return tf.reduce_sum(diff, axis=2)

    def initialize(self,
                   inputs_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   outputs_type=tf.float32,
                   reset=True,
                   **kwargs):
        """Initialize model.

        Arguments:
            inputs_shape -- shape of inputs layer
            outputs_shape -- shape of outputs layer
            inputs_type -- type of inputs layer
            outputs_type -- type of outputs layer
            reset -- indicator of clearing default graph and logging directory
            kwargs -- dictionary of keyword arguments

        """
        super(TFEmbedding, self).initialize(inputs_shape=inputs_shape,
                                            targets_shape=[],
                                            outputs_shape=outputs_shape,
                                            inputs_type=inputs_type,
                                            targets_type=tf.int32,
                                            outputs_type=outputs_type,
                                            reset=reset,
                                            **kwargs)

        def centroid_dist(embedding_pos, embedding_neg):
            """Centroid distances."""
            centroid = tf.reduce_mean(embedding_pos, 0)
            print embedding_pos
            print embedding_neg
            print centroid
            centroid_pos_dist = TFEmbedding.squared_distance(
                tf.expand_dims(centroid, 0),
                embedding_pos)
            centroid_neg_dist = TFEmbedding.squared_distance(
                tf.expand_dims(centroid, 0),
                embedding_neg)
            return centroid_pos_dist, centroid_neg_dist

        # Calculate distances
        embedding_pos, embedding_neg = tf.dynamic_partition(
            self.outputs,
            partitions=tf.reshape(self.targets, [-1]),
            num_partitions=2)
        centroid_pos_dist, centroid_neg_dist = centroid_dist(embedding_pos,
                                                             embedding_neg)

        # Add centroid distance metric
        self.add_metric('centroid_pos_dist',
                        centroid_pos_dist,
                        summary_type=tf.summary.histogram,
                        collections=['train', 'validation'])
        self.add_metric('centroid_neg_dist',
                        centroid_neg_dist,
                        summary_type=tf.summary.histogram,
                        collections=['train', 'validation'])

        def max_centroid_fscore(pos_dist, neg_dist):
            """Centroid binary classification fscore."""

            def fscore_function(threshold):
                tp = tf.cast(tf.count_nonzero(pos_dist < threshold),
                             tf.float32)
                fp = tf.cast(tf.count_nonzero(pos_dist >= threshold),
                             tf.float32)
                fn = tf.cast(tf.count_nonzero(neg_dist < threshold),
                             tf.float32)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                return 2. * (precision * recall) / (precision + recall)

            # Get all possible threshold values
            total_dist = tf.reshape(tf.concat([pos_dist, neg_dist], 1), [-1])
            thresholds = tf.unique(total_dist)[0]

            # Calculate fscores
            fscores = tf.map_fn(fscore_function, thresholds)
            return tf.reduce_max(fscores)

        # Add max centroid fscore metric
        self.add_metric('max_centroid_fscore',
                        max_centroid_fscore(centroid_pos_dist, centroid_neg_dist),
                        summary_type=tf.summary.scalar,
                        collections=['train', 'validation', 'log'])

    def loss_function(self, targets, outputs, **kwargs):
        """Compute the triplet loss by mini-batch of triplet embeddings.

        Arguments:
            targets -- tensor of batch with targets
            outputs -- tensor of batch with outputs
            kwargs -- dictionary of keyword arguments

        Return:
            loss -- triplet loss operation

        """
        assert 'margin' in kwargs, \
            '''Argument \'margin\' should be passed:
            kwargs = %s''' % kwargs

        assert 'exclude_hard' in kwargs, \
            '''Argument \'exclude_hard\' should be passed:
            kwargs = %s''' % kwargs

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
        embedding_pos, embedding_neg = tf.dynamic_partition(
            outputs,
            partitions=self.targets,
            num_partitions=2)
        pos_dist = TFEmbedding.squared_distance(embedding_pos, embedding_pos)
        neg_dist = TFEmbedding.squared_distance(embedding_pos, embedding_neg)

        # Calculate losses
        losses = tf.map_fn(fn=triplet_loss(margin, exclude_hard),
                           elems=(pos_dist, neg_dist),
                           dtype=tf.float32)
        return tf.reduce_mean(losses)

    @check_initialization
    def fit(self,
            train_set,
            epoch_count=None,
            iter_count=None,
            optimizer=tf.train.RMSPropOptimizer,
            learning_rate=0.001,
            val_set=None,
            summarizing_period=1,
            logging_period=100,
            checkpoint_period=10000,
            evaluation_period=10000,
            max_gradient_norm=None):
        """Train model.

        Arguments:
            train_set -- dataset for training
            epoch_count -- training epochs count
            iter_count -- training iterations count
            optimizer -- tensorflow optimizer object
            learning_rate -- initial gradient descent step
            val_set -- dataset for validation
            summarizing_period -- iterations count between summarizing
            logging_period -- iterations count between logging to stdout
            checkpoint_period -- iterations count between saving checkpoint
            evaluation_period -- iterations count between evaluation

        """
        assert isinstance(train_set, TFTripletset), \
            '''Training set should be object of TFTripletset type:
            type(train_set) = %s''' % type(train_set)
        if val_set is not None:
            assert isinstance(val_set, TFTripletset), \
                '''Validation set should be object of TFTripletset type:
                type(val_set) = %s''' % type(val_set)

        super(TFEmbedding, self).fit(train_set=train_set,
                                     epoch_count=epoch_count,
                                     iter_count=iter_count,
                                     optimizer=optimizer,
                                     learning_rate=learning_rate,
                                     val_set=val_set,
                                     summarizing_period=summarizing_period,
                                     logging_period=logging_period,
                                     checkpoint_period=checkpoint_period,
                                     evaluation_period=evaluation_period,
                                     max_gradient_norm=max_gradient_norm)

    @check_initialization
    def evaluate(self, data, collection='eval'):
        """Evaluate model.

        Arguments:
            data -- batch or dataset of inputs
            collection -- string value from ['train', 'validation', 'eval']

        Return:
            result -- metrics dictionary

        """
        if isinstance(data, TFBatch):
            return super(TFEmbedding, self).evaluate(data, collection=collection)
        else:
            warnings.warn('''Evaluation function is not implemented for type:
                type(data) = %s''' % type(data), Warning)
            return {}

    @check_initialization
    @check_inputs_values
    def visualize(self, inputs_values, var_name, labels=None):
        """Visualize embeddings in TensorBoard.

        Arguments:
            inputs_values -- batch of inputs
            var_name -- string key
            labels -- optional embeddings labels

        """
        if labels is not None:
            assert len(inputs_values) == len(labels), \
                '''Inputs values and labels should be the same lengths:
                len(inputs_values) = %s, len(labels) = %s''' \
                % (len(inputs_values), len(labels))

        print('Visualizing...')

        # Get visualization embeddings
        vis_embeddings = self.forward(inputs_values)
        if labels is not None:
            vis_labels = labels.flatten()
        vis_name = tf.get_default_graph().unique_name(var_name,
                                                      mark_as_used=False)

        # Input set for Embedded TensorBoard visualization
        vis_var = tf.Variable(tf.stack(vis_embeddings, axis=0),
                              trainable=False,
                              name=vis_name)
        self.sess.run(tf.variables_initializer([vis_var]))

        # Add embedding tensorboard visualization
        embed = self.projector_config.embeddings.add()
        embed.tensor_name = vis_name + ':0'
        if labels is not None:
            embed.metadata_path = os.path.join(
                self.log_dir,
                embed.tensor_name + '_metadata.tsv')
        projector.visualize_embeddings(self.summary_writer,
                                       self.projector_config)

        # Checkpoint configuration
        checkpoint_name = 'vis-checkpoint'
        checkpoint_file = os.path.join(self.log_dir, checkpoint_name)

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
        print('''For watching in TensorBoard run command:
              tensorboard --logdir "%s"''' % self.log_dir)
