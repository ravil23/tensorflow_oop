import argparse
import os
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Include additional module
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, '..')
if include_dir not in sys.path:
    sys.path.append(include_dir)
from tensorflow_oop.embedding import *

# Define model
class MnistEmbedding(TFEmbedding):
    def inference(self, inputs, **kwargs):
        input_size = self.inputs_shape[0]
        hidden_size = kwargs['hidden_size']
        output_size = self.outputs_shape[0]
        with tf.name_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal([input_size, hidden_size],
                                    stddev=1.0 / np.sqrt(float(input_size))))
            biases = tf.Variable(tf.zeros([hidden_size]))
            hidden = tf.nn.relu(tf.nn.xw_plus_b(inputs, weights, biases))
        with tf.name_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([hidden_size, output_size],
                                    stddev=1.0 / np.sqrt(float(hidden_size))))
            biases = tf.Variable(tf.zeros([output_size]))
            outputs = tf.nn.xw_plus_b(hidden, weights, biases)
        embeddings = tf.nn.l2_normalize(outputs, 1)
        return embeddings

def run(args):
    print('Loading dataset...')
    mnist = input_data.read_data_sets(args.input)
    train_set = TFTripletset(mnist.train.images, mnist.train.labels)
    train_set.set_batch_size(args.batch_size, args.batch_positives_count)
    val_set = TFTripletset(mnist.validation.images, mnist.validation.labels)
    val_set.set_batch_size(args.batch_size, args.batch_positives_count)
    test_set = TFTripletset(mnist.test.images, mnist.test.labels)
    test_set.set_batch_size(args.batch_size, args.batch_positives_count)
    print('Traininig  set shape: %s' % train_set.str_shape())
    print('Validation set shape: %s' % val_set.str_shape())
    print('Testing    set shape: %s\n' % test_set.str_shape())

    # Initialization
    model = MnistEmbedding(log_dir=args.log_dir)
    if args.load:
        model.load()
    else:
        model.initialize(inputs_shape=train_set.data_shape,
                         outputs_shape=[args.embedding_size],
                         hidden_size=args.hidden_size,
                         margin=args.margin,
                         exclude_hard=args.exclude_hard)
    print('%s\n' % model)

    # Fitting model
    model.fit(train_set, epoch_count=args.epoch_count, val_set=val_set)

    # Calculate visualisation embeddings and labels
    vis_train_embeddings = model.forward(train_set.data[:args.vis_count])
    vis_train_labels = train_set.labels[:args.vis_count]
    vis_val_embeddings = model.forward(val_set.data[:args.vis_count])
    vis_val_labels = val_set.labels[:args.vis_count]
    vis_test_embeddings = model.forward(test_set.data[:args.vis_count])
    vis_test_labels = test_set.labels[:args.vis_count]

    # Visualization
    model.visualize(vis_train_embeddings, 'train_set', labels=vis_train_labels)
    model.visualize(vis_val_embeddings, 'val_set', labels=vis_val_labels)
    model.visualize(vis_test_embeddings, 'test_set', labels=vis_test_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MNIST embedding with Triplet Loss.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', default='mnist_data', type=str,
        help='path to directory with input data archives')
    parser.add_argument('--log_dir', default='/tmp/mnist_emb', type=str,
        help='path to directory for logging')
    parser.add_argument('--batch_size', default=32, type=int,
        help='batch size')
    parser.add_argument('--batch_positives_count', default=4, type=int,
        help='batch positives count')
    parser.add_argument('--hidden_size', default=64, type=int,
        help='hidden layer size')
    parser.add_argument('--embedding_size', default=3, type=int,
        help='embedding layer size')
    parser.add_argument('--epoch_count', default=2, type=int,
        help='training epoch count')
    parser.add_argument('--vis_count', default=5000, type=int,
        help='visualization elements count')
    parser.add_argument('--margin', default=0.5, type=float,
        help='triplet loss margin')    
    parser.add_argument('--exclude_hard', default=False, action='store_true',
        help='exclude negative elements with loss greater than margin')
    parser.add_argument('--load', default=False, action='store_true',
        help='load from checkpoint')

    args = parser.parse_args()

    run(args)
