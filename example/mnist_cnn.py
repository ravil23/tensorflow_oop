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
class MnistCNN(TFClassifier):
    def inference(self, inputs, **kwargs):
        input_size = self.inputs_shape_[0]
        hidden_size = kwargs['hidden_size']
        output_size = self.outputs_shape_[0]
        inputs = tf.reshape(inputs, shape=[-1, 28, 28, 1])
        with tf.variable_scope('conv_1'):
            conv_1 = self.conv2d(inputs, [5, 5, 1, 32], [32])
            pool_1 = self.max_pool(conv_1)
        with tf.variable_scope('conv_2'):
            conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64])
            pool_2 = self.max_pool(conv_2)
        with tf.variable_scope('fc'):
            pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
            fc_1 = self.fc(pool_2_flat, [7*7*64, hidden_size], [hidden_size])
        with tf.variable_scope('output'):
            outputs = self.fc(fc_1, [hidden_size, output_size], [output_size])
        return outputs

    def conv2d(self, inputs, weight_shape, bias_shape):
        incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_init = tf.random_normal_initializer(stddev=(2.0/incoming)**0.5)
        W = tf.get_variable('W', weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value=0)
        b = tf.get_variable('b', bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding='SAME'), b))

    def max_pool(self, inputs, k=2):
        return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def fc(self, inputs, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable('W', weight_shape, initializer=weight_init)
        b = tf.get_variable('b', bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.matmul(inputs, W) + b)

def run(args):
    print('Loading dataset...')
    mnist = input_data.read_data_sets(args.input, one_hot=True)
    train_set = TFDataset(mnist.train.images, mnist.train.labels)
    train_set.set_batch_size(args.batch_size)
    val_set = TFDataset(mnist.validation.images, mnist.validation.labels)
    val_set.set_batch_size(args.batch_size)
    test_set = TFDataset(mnist.test.images, mnist.test.labels)
    test_set.set_batch_size(args.batch_size)
    print('Traininig  set shape:', train_set.size_, train_set.data_shape_, '->', train_set.labels_shape_)
    print('Validation set shape:', val_set.size_, val_set.data_shape_, '->', val_set.labels_shape_)
    print('Testing    set shape:', test_set.size_, test_set.data_shape_, '->', test_set.labels_shape_, '\n')

    print('Initializing...')
    model = MnistCNN(log_dir=args.log_dir,
                     inputs_shape=train_set.data_shape_,
                     outputs_shape=train_set.labels_shape_,
                     hidden_size=args.hidden_size)
    print(model, '\n')

    # Fitting model
    model.fit(train_set, iteration_count=None, epoch_count=args.epoch_count, val_set=val_set)
    
    print('Evaluating...')
    if train_set is not None:
        train_eval = model.evaluate(train_set)
        print('Results on training set:', train_eval)
    if val_set is not None:
        val_eval = model.evaluate(val_set)
        print('Results on validation set:', val_eval)
    if test_set is not None:
        test_eval = model.evaluate(test_set)
        print('Results on testing set:', test_eval)

    if args.show:
        print('Showing test set...')
        import cv2
        def show(data):
            cv2.imshow('image', data.reshape((28,28,1)))
            cv2.waitKey()
        for i in range(test_set.size_):
            result = model.classify([test_set.data_[i]])
            print('Classification result:', result)
            show(test_set.data_[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST classification with Convolutional neural network (CNN).',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', default='mnist_data', type=str,
        help='path to directory with input data archives')
    parser.add_argument('--log_dir', default='/tmp/mnist_cnn', type=str,
        help='path to directory for logging')
    parser.add_argument('--batch_size', default=32, type=int,
        help='batch size')
    parser.add_argument('--hidden_size', default=64, type=int,
        help='hidden layer size')
    parser.add_argument('--epoch_count', default=2, type=int,
        help='training epoch count')
    parser.add_argument('--show', default=False, action='store_true',
        help='show classification results in window (OpenCV requiered)')

    args = parser.parse_args()

    run(args)
