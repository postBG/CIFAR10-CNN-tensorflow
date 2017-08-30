import unittest

from layers import input_image, label
from utils import Cifar10


class TestLayers(unittest.TestCase):
    def test_input_image(self):
        image_shape = (32, 32, 3)
        input_tensor = input_image(image_shape, name="input")

        assert input_tensor.get_shape().as_list() == [None, image_shape[0], image_shape[1], image_shape[2]], \
            'Incorrect Image Shape.  Found {} shape'.format(input_tensor.get_shape().as_list())

        assert input_tensor.op.type == 'Placeholder', \
            'Incorrect Image Type.  Found {} type'.format(input_tensor.op.type)

        assert input_tensor.name == 'input:0', \
            'Incorrect Name.  Found {}'.format(input_tensor.name)

        print('Image Input Tests Passed.')

    def test_label(self):
        lable_tensor = label(name="y")

        assert lable_tensor.get_shape().as_list() == [None, Cifar10.num_classes], \
            'Incorrect Label Shape.  Found {} shape'.format(lable_tensor.get_shape().as_list())

        assert lable_tensor.op.type == 'Placeholder', \
            'Incorrect Label Type.  Found {} type'.format(lable_tensor.op.type)

        assert lable_tensor.name == 'y:0', \
            'Incorrect Name.  Found {}'.format(lable_tensor.name)

        print('Label Input Tests Passed.')
