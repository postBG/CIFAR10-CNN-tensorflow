import tensorflow as tf

from data import DataManager


def _optimizer(model):
    return tf.train.AdamOptimizer().minimize(model.loss)


def _correct_pred(model):
    return tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.labels, 1))


def _accuracy(model):
    return tf.reduce_mean(tf.cast(_correct_pred(model), tf.float32), name="accuracy")


class Trainer(object):
    def __init__(self, model, inputs, data_manager, config):
        self.validation_features, self.validation_labels = DataManager.load_preprocess_validation()

        self.optimizer = _optimizer(model)
        self.correct_pred = _correct_pred(model)
        self.accuracy = _accuracy(model)

        self.model = model
        self.inputs = inputs
        self.data_manager = data_manager

        self.batch_size = config.batch_size
        self.config = config

    def run(self, session):
        for epoch in range(self.config.epochs):
            self.train_one_epoch(session, epoch=epoch)

    def train_one_epoch(self, session, *args, **kwargs):
        for batch_id in range(1, 2):
            for features, labels in self.data_manager.load_preprocess_training_batch(batch_id, self.batch_size):
                session.run(self.optimizer, feed_dict={
                    self.inputs: features,
                    self.model.labels: labels,
                    self.model.dropout_rate: self.config.dropout_rate
                })

            loss, accur = self.validate_stats(session)
            self._print_stats(loss, accur, batch_i=batch_id, epoch=kwargs.get('epoch'))

    def validate_stats(self, session):
        loss = session.run(self.model.loss, feed_dict={
            self.inputs: self.validation_features,
            self.model.labels: self.validation_labels,
            self.model.dropout_rate: 1.0
        })

        accur = session.run(self.accuracy, feed_dict={
            self.inputs: self.validation_features,
            self.model.labels: self.validation_labels,
            self.model.dropout_rate: 1.0
        })

        return loss, accur

    def _print_stats(self, loss, accur, *args, **kwargs):
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(kwargs.get('epoch', 0), kwargs.get('batch_i')), end='')
        print('Traning Loss: {:>10.4f} Accuracy: {:.6f}'.format(loss, accur))
