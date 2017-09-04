import tensorflow as tf
import unittest

from layers import input_image, label, fully_connected, flatten, conv2d, max_pool2d
from utils import Cifar10


class TestLayers(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_input_image(self):
        image_shape = (32, 32, 3)
        input_tensor = input_image(image_shape, name="input")

        self.assertEquals([None, *image_shape], input_tensor.get_shape().as_list(),
                          'Incorrect Image Shape.  Found {} shape'.format(input_tensor.get_shape().as_list()))

        self.assertEquals('Placeholder', input_tensor.op.type,
                          'Incorrect Image Type.  Found {} type'.format(input_tensor.op.type))

        self.assertEquals('input:0', input_tensor.name,
                          'Incorrect Name.  Found {}'.format(input_tensor.name))

        print('Image Input Tests Passed.')

    def test_label(self):
        lable_tensor = label(name="y")

        self.assertEquals([None, Cifar10.num_classes], lable_tensor.get_shape().as_list(),
                          'Incorrect Label Shape.  Found {} shape'.format(lable_tensor.get_shape().as_list()))

        self.assertEquals('Placeholder', lable_tensor.op.type,
                          'Incorrect Label Type.  Found {} type'.format(lable_tensor.op.type))

        self.assertEquals('y:0', lable_tensor.name,
                          'Incorrect Name.  Found {}'.format(lable_tensor.name))

        print('Label Input Tests Passed.')

    def test_fully_conn(self):
        test_x = tf.placeholder(tf.float32, [None, 128])
        test_num_outputs = 40

        fc_out = fully_connected(test_x, test_num_outputs)

        self.assertEquals([None, 40], fc_out.get_shape().as_list(),
                          msg='Incorrect Shape.  Found {} shape'.format(fc_out.get_shape().as_list()))

        print('Fully Connected Tests Passed.')

    def test_flatten(self):
        test_x = tf.placeholder(tf.float32, [None, 12, 12, 64])

        flat = flatten(test_x)

        self.assertEquals([None, 12 * 12 * 64], flat.get_shape().as_list(),
                          msg='Incorrect Shape.  Found {} shape'.format(flat.get_shape().as_list()))

        print('Flatten Tests Passed.')

    def test_conv2d(self):
        test_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        num_filters = 10

        conv = conv2d(test_x, num_filters, kernel_size=[2, 2], stride=[2, 2])

        self.assertEquals([None, 16, 16, 10], conv.get_shape().as_list(),
                          msg='Incorrect Shape.  Found {} shape'.format(conv.get_shape().as_list()))

        print('Conv2d Tests Passed.')

    def test_max_pool2d(self):
        test_x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        kernel_size = [2, 2]

        max_pool = max_pool2d(test_x, kernel_size)

        self.assertEquals([None, 32, 32, 3], max_pool.get_shape().as_list(),
                          msg='Incorrect Shape.  Found {} shape'.format(max_pool.get_shape().as_list()))

        print('Max Pool 2d Tests Passes')
