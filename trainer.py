import tensorflow as tf


def optimizer(model):
    return tf.train.AdamOptimizer().minimize(model.loss)


def correct_pred(model):
    return tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.labels, 1))


def accuracy(model):
    return tf.reduce_mean(tf.cast(correct_pred(model), tf.float32), name="accuracy")
