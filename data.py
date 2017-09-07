import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer

from utils import Cifar10, unpickle

preprocessed_data_path = "preprocessed-cifar10-data"

lb = LabelBinarizer()
lb.fit(range(Cifar10.num_classes))

if not os.path.exists(preprocessed_data_path):
    os.makedirs(preprocessed_data_path)


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    return lb.transform(x)


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    used min-max normalization

    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    max_value = 255
    min_value = 0
    return (x - min_value) / (max_value - min_value)


class DataManager(Cifar10):
    @staticmethod
    def _get_preprocessed_filename(filename):
        return os.path.join(preprocessed_data_path, filename)

    @staticmethod
    def _get_processed_train_batch_filename(batch_id):
        return DataManager._get_preprocessed_filename('preprocess_batch_' + str(batch_id) + '.p')

    @staticmethod
    def _preprocess_and_save(features_preprocessor, labels_preprocessor, features, labels, filename):
        """
        Preprocess data and save it to file
        """
        processed_features = features_preprocessor(features)
        processed_labels = labels_preprocessor(labels)

        pickle.dump((processed_features, processed_labels), open(filename, 'wb'))

    def preprocess_and_save(self, features_preprocessor, labels_preprocessor, validation_rate=0.1):
        n_batches = 5
        validation_features = []
        validation_labels = []

        for batch_id in range(1, n_batches + 1):
            features, labels = self.load_batch(batch_id)
            validation_count = int(len(features) * validation_rate)

            # Prprocess and save a batch of training data
            DataManager._preprocess_and_save(
                features_preprocessor,
                labels_preprocessor,
                features[:-validation_count],
                labels[:-validation_count],
                DataManager._get_processed_train_batch_filename(batch_id))

            # Use a portion of training batch for validation
            validation_features.extend(features[-validation_count:])
            validation_labels.extend(labels[-validation_count:])

        # Preprocess and Save all validation data
        DataManager._preprocess_and_save(
            features_preprocessor,
            labels_preprocessor,
            np.array(validation_features),
            np.array(validation_labels),
            os.path.join(preprocessed_data_path, 'preprocess_validation.p'))

        test_batch = unpickle('test_batch')

        # load the training data
        test_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = test_batch['labels']

        # Preprocess and Save all training data
        DataManager._preprocess_and_save(
            features_preprocessor,
            labels_preprocessor,
            np.array(test_features),
            np.array(test_labels),
            os.path.join(preprocessed_data_path, 'preprocess_test.p'))

    @staticmethod
    def batch_features_labels(features, labels, batch_size):
        """
        Split features and labels into batches
        """
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

    @staticmethod
    def load_preprocess_training_batch(batch_id, batch_size):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        filename = DataManager._get_processed_train_batch_filename(batch_id)
        features, labels = pickle.load(open(filename, mode='rb'))

        return DataManager.batch_features_labels(features, labels, batch_size)

    @staticmethod
    def load_preprocess_validation():
        validation_filepath = DataManager._get_preprocessed_filename('preprocess_validation.p')
        return pickle.load(open(validation_filepath, mode='rb'))

    @staticmethod
    def load_preprocess_test():
        test_filepath = DataManager._get_preprocessed_filename('preprocess_test.p')
        return pickle.load(open(test_filepath, mode='rb'))
