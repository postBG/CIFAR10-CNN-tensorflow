import tensorflow as tf

from layers import conv2d, fully_connected, max_pool2d, flatten, dropout, label
from utils import Cifar10


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def linear(x):
    return x


activation_fn_dict = {
    'linear': linear,
    'relu': tf.nn.relu,
    'leaky': leaky_relu
}


class SimpleCNN(object):
    def __init__(self, config, images):
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_strides = config.conv_strides
        self.activation_fn = activation_fn_dict[config.activation_fn]
        self.config = config

        self.labels = label('y')
        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.build_graph(images)

    def build_graph(self, images):
        self.conv1 = conv2d(images, 32, self.conv_kernel_size, self.conv_strides, self.activation_fn, name='conv1')
        self.pool1 = max_pool2d(self.conv1, name='pool1')

        self.conv2 = conv2d(self.pool1, 64, self.conv_kernel_size, self.conv_strides, self.activation_fn, name='conv2')
        self.pool2 = max_pool2d(self.conv2, name='pool2')

        self.conv3 = conv2d(self.pool2, 128, self.conv_kernel_size, self.conv_strides, self.activation_fn, name='conv3')
        self.pool3 = max_pool2d(self.conv3, name='pool3')

        self.conv4 = conv2d(self.pool3, 256, self.conv_kernel_size, self.conv_strides, self.activation_fn, name='conv4')
        self.pool4 = max_pool2d(self.conv4, name='pool4')

        self.flatten = flatten(self.pool4)

        with tf.name_scope('fc1'):
            self.fc1 = dropout(fully_connected(self.flatten, 512, activation_fn=self.activation_fn), self.dropout_rate)
        with tf.name_scope('fc2'):
            self.fc2 = dropout(fully_connected(self.fc1, 256, activation_fn=self.activation_fn), self.dropout_rate)
        with tf.name_scope('fc3'):
            self.fc3 = dropout(fully_connected(self.fc2, 128, activation_fn=self.activation_fn), self.dropout_rate)

        self.logits = fully_connected(self.fc3, Cifar10.num_classes, name="logits")
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
