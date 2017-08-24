from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

from tests import test_folder_path, test_cifar10_tar_data_exists

cifar10_dataset_folder_path = 'cifar-10-batches-py'
floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
if isfile(floyd_cifar10_location):
    tar_gz_path = floyd_cifar10_location
else:
    tar_gz_path = 'cifar-10-python.tar.gz'


class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_cifar10_tar():
    if not isfile(tar_gz_path):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_gz_path,
                pbar.hook)

    test_cifar10_tar_data_exists(tar_gz_path)


def unzip():
    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall()
            tar.close()


def download_and_unzip():
    download_cifar10_tar()
    unzip()
    test_folder_path(cifar10_dataset_folder_path)
