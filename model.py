import tensorflow as tf

from layers import conv2d, fully_connected, max_pool2d, flatten, dropout, label
from utils import Cifar10


class NeuralNetworkModel(object):
    def loss(self):
        raise NotImplementedError()


class SimpleCNN(NeuralNetworkModel):
    def __init__(self, config, images):
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_strides = config.conv_strides
        self.activation_fn = config.activation_fn
        self.dropout_rate = config.dropout_rate
        self.config = config

        self.build_graph(images)
        self.labels = label('y')

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

    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
