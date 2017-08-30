import unittest
import numpy as np

from data import normalize, one_hot_encode


def _print_success_message():
    return print('Tests Passed')


class TestUtils(unittest.TestCase):
    def test_normalize(self):
        test_shape = (np.random.choice(range(1000)), 32, 32, 3)
        test_numbers = np.random.choice(range(256), test_shape)
        normalize_out = normalize(test_numbers)

        assert type(normalize_out).__module__ == np.__name__, \
            'Not Numpy Object'

        assert normalize_out.shape == test_shape, \
            'Incorrect Shape. {} shape found'.format(normalize_out.shape)

        assert normalize_out.max() <= 1 and normalize_out.min() >= 0, \
            'Incorect Range. {} to {} found'.format(normalize_out.min(), normalize_out.max())

        _print_success_message()

    def test_one_hot_encode(self):
        test_shape = np.random.choice(range(1000))
        test_numbers = np.random.choice(range(10), test_shape)
        one_hot_out = one_hot_encode(test_numbers)

        assert type(one_hot_out).__module__ == np.__name__, \
            'Not Numpy Object'

        assert one_hot_out.shape == (test_shape, 10), \
            'Incorrect Shape. {} shape found'.format(one_hot_out.shape)

        n_encode_tests = 5
        test_pairs = list(zip(test_numbers, one_hot_out))
        test_indices = np.random.choice(len(test_numbers), n_encode_tests)
        labels = [test_pairs[test_i][0] for test_i in test_indices]
        enc_labels = np.array([test_pairs[test_i][1] for test_i in test_indices])
        new_enc_labels = one_hot_encode(labels)

        assert np.array_equal(enc_labels, new_enc_labels), \
            'Encodings returned different results for the same numbers.\n' \
            'For the first call it returned:\n' \
            '{}\n' \
            'For the second call it returned\n' \
            '{}\n' \
            'Make sure you save the map of labels to encodings outside of the function.'.format(enc_labels,
                                                                                                new_enc_labels)

        for one_hot in new_enc_labels:
            assert (one_hot == 1).sum() == 1, \
                'Each one-hot-encoded value should include the number 1 exactly once.\n' \
                'Found {}\n'.format(one_hot)
            assert (one_hot == 0).sum() == len(one_hot) - 1, \
                'Each one-hot-encoded value should include zeros in all but one position.\n' \
                'Found {}\n'.format(one_hot)

        _print_success_message()
