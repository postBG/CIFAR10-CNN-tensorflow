import tensorflow as tf

import trainer
from download import download_and_unzip
from data import one_hot_encode, normalize, DataManager
from model import SimpleCNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('conv_kernel_size', 3,
                            """Convolution Layers kernel size.""")
tf.app.flags.DEFINE_integer('conv_strides', 3,
                            """Convolution Layers stride size""")
tf.app.flags.DEFINE_integer('epochs', 10,
                            """Number of epochs""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Size of batches""")
tf.app.flags.DEFINE_float('dropout_rate', 0.5,
                          """Dropout rate. 0.1 would drop out 10% of input units.""")
tf.app.flags.DEFINE_string('activation_fn', 'relu',
                           """One of [linear, relu, leaky]""")
tf.app.flags.DEFINE_boolean('need_preprocess', True,
                            """If you already preprocess and save the data, then Set this flags to False""")


def main(argv=None):
    download_and_unzip()

    data_manager = DataManager()

    if(FLAGS.need_preprocess):
        data_manager.preprocess_and_save(normalize, one_hot_encode)

    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
    net = SimpleCNN(FLAGS, inputs)

    model_trainer = trainer.Trainer(net)
    validation_features, validation_labels = DataManager.load_preprocess_validation()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            batch_i = 1
            for features, labels in data_manager.load_preprocess_training_batch(batch_i, FLAGS.batch_size):
                sess.run(model_trainer.optimizer, feed_dict={
                    inputs: features,
                    net.labels: labels,
                    net.dropout_rate: FLAGS.dropout_rate
                })

            loss = sess.run(net.loss, feed_dict={
                inputs: validation_features,
                net.labels: validation_labels,
                net.dropout_rate: 1.0
            })

            accur = sess.run(model_trainer.accuracy, feed_dict={
                inputs: validation_features,
                net.labels: validation_labels,
                net.dropout_rate: 1.0
            })
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print('Traning Loss: {:>10.4f} Accuracy: {:.6f}'.format(loss, accur))


if __name__ == '__main__':
    tf.app.run()
