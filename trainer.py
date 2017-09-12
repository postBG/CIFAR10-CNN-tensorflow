import tensorflow as tf

from data import DataManager


def _optimizer(model):
    return tf.train.AdamOptimizer().minimize(model.loss)


def _correct_pred(model):
    return tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.labels, 1))


def _accuracy(model):
    return tf.reduce_mean(tf.cast(_correct_pred(model), tf.float32), name="accuracy")


class Trainer(object):
    def __init__(self, model):
        self.validation_features, self.validation_labels = DataManager.load_preprocess_validation()

        self.optimizer = _optimizer(model)
        self.correct_pred = _correct_pred(model)
        self.accuracy = _accuracy(model)

        self.model = model

