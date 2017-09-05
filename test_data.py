import unittest
import numpy as np

from data import normalize, one_hot_encode


class TestUtils(unittest.TestCase):
    def test_normalize(self):
        test_shape = (np.random.choice(range(1000)), 32, 32, 3)
        test_numbers = np.random.choice(range(256), test_shape)
        normalize_out = normalize(test_numbers)

        self.assertEquals(np.__name__, type(normalize_out).__module__)
        self.assertEquals(test_shape, normalize_out.shape)
        self.assertTrue(normalize_out.max() <= 1 and normalize_out.min() >= 0)

    def test_one_hot_encode(self):
        test_shape = np.random.choice(range(1000))
        test_numbers = np.random.choice(range(10), test_shape)
        one_hot_out = one_hot_encode(test_numbers)

        self.assertEquals(np.__name__, type(one_hot_out).__module__)
        self.assertEquals((test_shape, 10), one_hot_out.shape)

        n_encode_tests = 5
        test_pairs = list(zip(test_numbers, one_hot_out))
        test_indices = np.random.choice(len(test_numbers), n_encode_tests)
        labels = [test_pairs[test_i][0] for test_i in test_indices]
        enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
        new_enc_labels = one_hot_encode(labels)

        self.assertTrue(np.array_equal(enc_labels, new_enc_labels))

        for one_hot in new_enc_labels:
            self.assertEquals(1, (one_hot == 1).sum())
            self.assertEquals(len(one_hot) - 1, (one_hot == 0).sum())
