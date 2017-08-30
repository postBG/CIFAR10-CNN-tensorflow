import tensorflow as tf

from utils import Cifar10


def input_image(image_shape, name):
    return tf.placeholder(tf.float32, shape=[None, *image_shape], name=name)


def label(name):
    return tf.placeholder(tf.float32, shape=[None, Cifar10.num_classes], name=name)


def conv2d(inputs, num_filters, kernel_size, stride, activation_fn=None, padding='same',
           weights_initializer=tf.contrib.layers.xavier_initializer(), name=None, **kwargs):
    return tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=stride,
                            activation=activation_fn, padding=padding,
                            kernel_initializer=weights_initializer, name=name, **kwargs)


def max_pool2d(inputs, kernel_size=(3, 3), stride=(1, 1), padding='same', name=None, **kwargs):
    return tf.layers.max_pooling2d(inputs, pool_size=kernel_size, strides=stride,
                                   padding=padding, name=name, **kwargs)


def flatten(inputs):
    return tf.contrib.layers.flatten(inputs)


def fully_connected(inputs, num_outputs, activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer(), name=None, **kwargs):
    return tf.layers.dense(inputs, units=num_outputs, activation=activation_fn,
                           kernel_initializer=weights_initializer, name=name, **kwargs)


def dropout(inputs, dropout_rate):
    return tf.layers.dropout(inputs, rate=dropout_rate)
