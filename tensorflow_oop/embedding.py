"""
Embedding base models.
"""

from tensorflow_oop.neural_network import *
from tensorflow_oop.decorators import *


class TFTripletset(TFDataset):

    """
    Triplet generation dataset.

    Attributes:
        ...                         Parrent class atributes.
        batch_positives_count       Positive elements count per batch.
        batch_negatives_count       Negative elements count per batch.
    """

    __slots__ = TFDataset.__slots__ + ['batch_positives_count',
                                       'batch_negatives_count']

    @check_triplets_data_labels
    def initialize(self, data, labels):
        """Set data and labels.

        Arguments:
            data        Array like object for store as data attribute.
            labels      Array like object for store as labels attribute.

        """
        super(TFTripletset, self).initialize(data=data, labels=labels)

    @check_initialization
    def set_batch_size(self, batch_size, batch_positives_count):
        """Set batch size and positives count per batch.

        Arguments:
            batch_size                Elements count per batch.
            batch_positives_count     Positive elements count per batch.

        """
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
    @update_last_batch
    def next_batch(self):
        """Get next batch."""
        labels = self.labels.flatten()
        labels_counts = np.bincount(labels)
        positive_keys = np.where(labels_counts >= self.batch_positives_count)[0]
        rand_pos_key = positive_keys[np.random.randint(0, len(positive_keys))]

        def random_sample(data, count):
            indexes = np.arange(len(data))
            rand_indexes = np.random.choice(indexes, count, replace=False)
            return data[rand_indexes], rand_indexes

        # Take positive samples
        positives, pos_indexes = random_sample(self.data[labels == rand_pos_key],
                                               self.batch_positives_count)

        # Take negative samples
        negatives, neg_indexes = random_sample(self.data[labels != rand_pos_key],
                                               self.batch_negatives_count)

        # Create batch
        batch_data = np.vstack([positives, negatives])
        batch_labels = np.append(np.zeros(len(positives)), np.ones(len(negatives)))
        batch_indexes = np.append(pos_indexes, neg_indexes)
        return TFBatch(data=batch_data, labels=batch_labels, indexes=batch_indexes)

class TFEmbedding(TFNeuralNetwork):

    """
    Embedding model with Triplet loss function.
    """

    @staticmethod
    def squared_distance(first_points, second_points):
        """Pairwise squared distances between 2 sets of points.

        Arguments:
            first_points       First collection of 1D tensors.
            second_points      Second collection of 1D tensors.

        """
        diff = tf.squared_difference(tf.expand_dims(first_points, 1),
                                     second_points)
        return tf.reduce_sum(diff, axis=2)

    def initialize(self,
                   inputs_shape,
                   outputs_shape,
                   inputs_type=tf.float32,
                   outputs_type=tf.float32,
                   print_self=True,
                   **kwargs):
        """Initialize model.

        Arguments:
            inputs_shape       Shape of inputs layer without batch dimension.
            outputs_shape      Shape of outputs layer without batch dimension.
            inputs_type        Type of inputs layer.
            outputs_type       Type of outputs layer.
            print_self         Indicator of printing model after initialization.
            kwargs             Dict object of options.

        """
        super(TFEmbedding, self).initialize(inputs_shape=inputs_shape,
                                            targets_shape=[],
                                            outputs_shape=outputs_shape,
                                            inputs_type=inputs_type,
                                            targets_type=tf.int32,
                                            outputs_type=outputs_type,
                                            print_self=False,
                                            **kwargs)

        def centroid_dist(embedding_pos, embedding_neg):
            """Centroid distances."""
            centroid = tf.reduce_mean(embedding_pos, 0)
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
        self.add_metric(tf.identity(centroid_pos_dist, 'centroid_pos_dist'),
                        collections=['batch_train'])
        self.add_metric(tf.identity(centroid_neg_dist, 'centroid_neg_dist'),
                        collections=['batch_train'])
        self.add_metric(tf.reduce_mean(centroid_pos_dist, name='mean_centroid_pos_dist'),
                        collections=['batch_train'])
        self.add_metric(tf.reduce_mean(centroid_neg_dist, name='mean_centroid_neg_dist'),
                        collections=['batch_train'])

        def max_fscore(pos_dist, neg_dist):
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
        max_centroid_fscore = max_fscore(centroid_pos_dist, centroid_neg_dist)
        self.add_metric(tf.identity(max_centroid_fscore, 'max_centroid_fscore'),
                        collections=['batch_train', 'batch_validation', 'log_train'])

        if print_self:
            print('%s\n' % self)

    def loss_function(self, targets, outputs):
        """Compute the triplet loss by mini-batch of triplet embeddings.

        Arguments:
            targets     Tensor of batch with targets.
            outputs     Tensor of batch with outputs.

        Return:
            loss        Triplet loss operation.

        """
        assert 'margin' in self.options, \
            '''Argument \'margin\' should be passed:
            self.options = %s''' % self.options

        assert 'exclude_hard' in self.options, \
            '''Argument \'exclude_hard\' should be passed:
            self.options = %s''' % self.options

        # Get options
        margin = self.options['margin']
        exclude_hard = self.options['exclude_hard']

        def triplet_loss(margin):
            """Triplet loss function for a given anchor."""
            def loss(pos_neg_dist):
                pos_dist, neg_dist = pos_neg_dist
                raw_loss = tf.expand_dims(pos_dist, -1) - neg_dist + margin
                mask = raw_loss > 0
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
        losses = tf.map_fn(fn=triplet_loss(margin),
                           elems=(pos_dist, neg_dist),
                           dtype=tf.float32)

        # Exclude hard losses if necessary
        if exclude_hard:
            valid_losses = tf.boolean_mask(losses, losses < margin)
        else:
            valid_losses = losses

        # Calculate some metrics
        out_mask = losses <= 0
        margin_mask = tf.logical_and(losses > 0, losses < margin)
        in_mask = losses >= margin

        # Add triplets count metric
        triplets_count_all = tf.cast(tf.size(losses), tf.float32, name='triplets_count_all')
        self.add_metric(triplets_count_all, collections=['batch_train'])

        triplets_portion_out = tf.divide(tf.count_nonzero(out_mask, dtype=tf.float32),
                                         triplets_count_all,
                                         name='triplets_portion_out')
        self.add_metric(triplets_portion_out, collections=['batch_train'])

        triplets_portion_margin = tf.divide(tf.count_nonzero(margin_mask, dtype=tf.float32),
                                            triplets_count_all,
                                            name='triplets_portion_margin')
        self.add_metric(triplets_portion_margin, collections=['batch_train'])

        triplets_portion_in = tf.divide(tf.count_nonzero(in_mask, dtype=tf.float32),
                                        triplets_count_all,
                                        name='triplets_portion_in')
        self.add_metric(triplets_portion_in, collections=['batch_train'])

        return tf.reduce_mean(valid_losses)

    @check_initialization
    def visualize(self, embeddings, var_name, labels=None):
        """Visualize embeddings in TensorBoard.

        Arguments:
            embeddings  Collection of embeddings.
            var_name    Name of variable in string format.
            labels      Optional embeddings labels with the same size.

        """
        if labels is not None:
            assert len(embeddings) == len(labels), \
                '''Embeddings and labels should be the same lengths:
                len(embeddings) = %s, len(labels) = %s''' \
                % (len(embeddings), len(labels))
            assert len(labels) == len(labels.flatten()), \
                '''Labels length should be equal to flatten labels:
                len(labels) = %s, len(labels.flatten()) = %s''' \
                % (len(labels), len(labels.flatten()))
        if len(embeddings) > 10000:
            warnings.warn('''Embeddings length should not be greater then 10000:
                          len(embeddings) = %s''' % len(embeddings))
            embeddings = embeddings[:10000]
            labels = labels[:10000]

        print('Visualizing...')

        # Get visualization embeddings
        vis_embeddings = embeddings
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
        embed = self._projector_config.embeddings.add()
        embed.tensor_name = vis_name + ':0'
        if labels is not None:
            embed.metadata_path = os.path.join(
                self.log_dir,
                embed.tensor_name + '_metadata.tsv')
        projector.visualize_embeddings(self._summary_writer,
                                       self._projector_config)

        # Save checkpoint
        tf.train.Saver(max_to_keep=1000).save(self.sess, self._vis_checkpoint)

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
