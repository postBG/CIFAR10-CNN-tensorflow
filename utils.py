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


def unpickle(filename):
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
    _meta = unpickle("batches.meta")
    num_classes = 10
    batch_ids = range(1, 6)

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

        batch = unpickle('data_batch_' + str(batch_id))
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
