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
class MnistRNN(TFClassifier):
    def inference(self, inputs):
        # Get options
        input_size = self.inputs_shape[0]
        hidden_size = self.options['hidden_size']
        output_size = self.outputs_shape[0]

        # Init stack of RNN cells
        def rnn_cell():
            return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        cells = [rnn_cell() for _ in range(2)]
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # Calculate dynamic batch size
        batch_size = tf.shape(inputs)[0]

        # Set initial state with zero
        init_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

        # Merge cells with dynamic RNN
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=multi_cell,
            initial_state=init_state,
            inputs=inputs,
            swap_memory=True,
            time_major=False)

        # Output fully connected layer
        with tf.variable_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([hidden_size, output_size],
                                    stddev=1.0 / np.sqrt(float(hidden_size))))
            biases = tf.Variable(tf.zeros([output_size]))
            outputs = tf.nn.xw_plus_b(rnn_outputs[:,-1], weights, biases)

        return outputs

def run(args):
    print('Loading dataset...')
    mnist = input_data.read_data_sets(args.input, one_hot=False)
    reshaped_data = np.reshape(mnist.train.images, [-1, 28, 28])
    train_set = TFSequence(reshaped_data, mnist.train.labels)
    train_set.set_batch_size(args.batch_size)
    reshaped_data = np.reshape(mnist.validation.images, [-1, 28, 28])
    val_set = TFSequence(reshaped_data, mnist.validation.labels)
    val_set.set_batch_size(args.batch_size)
    reshaped_data = np.reshape(mnist.test.images, [-1, 28, 28])
    test_set = TFSequence(reshaped_data, mnist.test.labels)
    test_set.set_batch_size(args.batch_size)
    print('Training   set shape: %s' % train_set.str_shape())
    print('Validation set shape: %s' % val_set.str_shape())
    print('Testing    set shape: %s\n' % test_set.str_shape())

    # Initialization
    model = MnistRNN(log_dir=args.log_dir, clear=not args.load)
    model.initialize(classes_count=10,
                     inputs_shape=train_set.data_shape,
                     outputs_shape=[10],
                     hidden_size=args.hidden_size)

    # Fitting model
    if args.load:
        model.restore()
    model.fit(train_set, val_set=val_set, epoch_count=args.epoch_count)

    # Evaluation
    if train_set is not None:
        model.evaluate_and_log(train_set, collection='eval_train')
    if val_set is not None:
        model.evaluate_and_log(val_set, collection='eval_validation')
    if test_set is not None:
        model.evaluate_and_log(test_set, collection='eval_test')

    # Print status info
    print('''For watching in TensorBoard run command:
          tensorboard --logdir "%s"''' % args.log_dir)

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
        description='MNIST classification with Recurent network (RNN).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', default='mnist_data', type=str,
        help='path to directory with input data archives')
    parser.add_argument('--log_dir', default='/tmp/mnist_rnn', type=str,
        help='path to directory for logging')
    parser.add_argument('--batch_size', default=32, type=int,
        help='batch size')
    parser.add_argument('--hidden_size', default=64, type=int,
        help='hidden layer size')
    parser.add_argument('--epoch_count', default=2, type=int,
        help='training epoch count')
    parser.add_argument('--load', default=False, action='store_true',
        help='load from checkpoint')
    parser.add_argument('--show', default=False, action='store_true',
        help='show classification results in window (OpenCV requiered)')

    args = parser.parse_args()

    run(args)
