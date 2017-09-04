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
from tensorflow_oop.classification import *

# Define model
class MnistFFN(TFClassifier):
    def inference(self, inputs, **kwargs):
        input_size = self.inputs_shape[0]
        hidden_size = kwargs['hidden_size']
        output_size = self.outputs_shape[0]
        with tf.variable_scope('hidden'):
            weights = tf.Variable(
                tf.truncated_normal([input_size, hidden_size],
                                    stddev=1.0 / np.sqrt(float(input_size))))
            biases = tf.Variable(tf.zeros([hidden_size]))
            hidden = tf.nn.relu(tf.nn.xw_plus_b(inputs, weights, biases))
        with tf.variable_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([hidden_size, output_size],
                                    stddev=1.0 / np.sqrt(float(hidden_size))))
            biases = tf.Variable(tf.zeros([output_size]))
            outputs = tf.nn.xw_plus_b(hidden, weights, biases)
        return outputs

def run(args):
    print('Loading dataset...')
    mnist = input_data.read_data_sets(args.input, one_hot=True)
    train_set = TFDataset(mnist.train.images, mnist.train.labels)
    val_set = TFDataset(mnist.validation.images, mnist.validation.labels)
    test_set = TFDataset(mnist.test.images, mnist.test.labels)
    print('Traininig  set shape: %s' % train_set.str_shape())
    print('Validation set shape: %s' % val_set.str_shape())
    print('Testing    set shape: %s\n' % test_set.str_shape())

    # Loading model
    model = TFClassifier(log_dir=args.log_dir)
    model.load()
    print('%s\n' % model)

    # Evaluation
    if train_set is not None:
        train_eval = model.evaluate(train_set)
        print('Results on training set: %s' % train_eval)
    if val_set is not None:
        val_eval = model.evaluate(val_set)
        print('Results on validation set: %s' % val_eval)
    if test_set is not None:
        test_eval = model.evaluate(test_set)
        print('Results on testing set: %s' % test_eval)

    if args.show:
        print('Showing test set...')
        import cv2
        def show(data):
            cv2.imshow('image', data.reshape((28,28,1)))
            cv2.waitKey()
        for i in range(test_set.size):
            result = model.predict([test_set.data[i]])
            print('Classification result: %s' % result)
            show(test_set.data[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MNIST classification model loading.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', default='mnist_data', type=str,
        help='path to directory with input data archives')
    parser.add_argument('--log_dir', default='/tmp/mnist_ffn', type=str,
        help='path to directory for logging')
    parser.add_argument('--show', default=False, action='store_true',
        help='show classification results in window (OpenCV requiered)')

    args = parser.parse_args()

    run(args)
