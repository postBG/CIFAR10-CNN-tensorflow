import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from download import cifar10_dataset_folder_path


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """
    return os.path.join(cifar10_dataset_folder_path, filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """
    file_path = _get_file_path(filename)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='latin1')

    return data


class Cifar10(object):
    def __init__(self):
        self._meta = _unpickle("batches.meta")
        self.num_classes = 10
        self.batch_ids = range(1, 6)

    @property
    def label_names(self):
        return self._meta['label_names']

    def label_name(self, label):
        return self.label_names[label]

    @property
    def images_per_file(self):
        return self._meta['num_cases_per_batch']

    def load_batch(self, batch_id):
        """
        Load a batch of the dataset
        
        :param batch_id: 로드할 batch의 id ([1, 5]) 
        :return: (features, labels)
        """
        if batch_id not in self.batch_ids:
            raise ValueError('Batch Id out of Range. Possible Batch Ids: {}'.format(self.batch_ids))

        batch = _unpickle('data_batch_' + str(batch_id))
        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        return features, labels

    def display_stats(self, batch_id, sample_idx=0):
        """
         Display Stats of the the dataset
         
        :param batch_id: 디스플레이할 batch의 id ([1, 5])  
        :param sample_idx: 디스플레이할 샘플의 index 
        """
        if batch_id not in self.batch_ids:
            raise ValueError('Batch Id out of Range. Possible Batch Ids: {}'.format(self.batch_ids))

        features, labels = self.load_batch(batch_id)

        if not (0 <= sample_idx < len(features)):
            raise ValueError('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_idx))

        print('\nStats of batch {}:'.format(batch_id))
        print('Samples: {}'.format(len(features)))
        print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
        print('First 20 Labels: {}'.format(labels[:20]))

        sample_image = features[sample_idx]
        sample_label = labels[sample_idx]

        print('\nExample of Image {}:'.format(sample_idx))
        print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
        print('Image - Shape: {}'.format(sample_image.shape))
        print('Label - Label Id: {} Name: {}'.format(sample_label, self.label_name(sample_label)))

        plt.axis('off')
        plt.imshow(sample_image)

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

        for batch_i in range(1, n_batches + 1):
            features, labels = self.load_batch(batch_i)
            validation_count = int(len(features) * validation_rate)

            # Prprocess and save a batch of training data
            Cifar10._preprocess_and_save(
                features_preprocessor,
                labels_preprocessor,
                features[:-validation_count],
                labels[:-validation_count],
                'preprocess_batch_' + str(batch_i) + '.p')

            # Use a portion of training batch for validation
            validation_features.extend(features[-validation_count:])
            validation_labels.extend(labels[-validation_count:])

        # Preprocess and Save all validation data
        Cifar10._preprocess_and_save(
            features_preprocessor,
            labels_preprocessor,
            np.array(validation_features),
            np.array(validation_labels),
            'preprocess_validation.p')

        test_batch = _unpickle('test_batch')

        # load the training data
        test_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = test_batch['labels']

        # Preprocess and Save all training data
        Cifar10._preprocess_and_save(
            features_preprocessor,
            labels_preprocessor,
            np.array(test_features),
            np.array(test_labels),
            'preprocess_training.p')

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
        filename = 'preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        # Return the training data in batches of size <batch_size> or less
        return Cifar10.batch_features_labels(features, labels, batch_size)
