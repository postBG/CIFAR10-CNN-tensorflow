import os


def test_cifar10_tar_data_exists(tar_gz_path):
    assert os.path.isfile(tar_gz_path), \
        'Cifar-10 tar file is not exists. Download it use download.download_cifar10_tar function.'


def test_folder_path(cifar10_dataset_folder_path):
    assert cifar10_dataset_folder_path is not None, \
        'Cifar-10 data folder not set.'
    assert cifar10_dataset_folder_path[-1] != '/', \
        'The "/" shouldn\'t be added to the end of the path.'
    assert os.path.exists(cifar10_dataset_folder_path), \
        'Path not found.'
    assert os.path.isdir(cifar10_dataset_folder_path), \
        '{} is not a folder.'.format(os.path.basename(cifar10_dataset_folder_path))

    train_files = [cifar10_dataset_folder_path + '/data_batch_' + str(batch_id) for batch_id in range(1, 6)]
    other_files = [cifar10_dataset_folder_path + '/batches.meta', cifar10_dataset_folder_path + '/test_batch']
    missing_files = [path for path in train_files + other_files if not os.path.exists(path)]

    assert not missing_files, \
        'Missing files in directory: {}'.format(missing_files)

    print('All files found!')
