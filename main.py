import tensorflow as tf

from download import download_and_unzip
from data import one_hot_encode, normalize, DataManager


def main():
    download_and_unzip()

    data_manager = DataManager()
    data_manager.preprocess_and_save(normalize, one_hot_encode)


if __name__ == '__main__':
    main()
