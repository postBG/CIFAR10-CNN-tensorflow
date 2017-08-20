import tensorflow as tf


def main():
    with tf.Session() as sess:
        print(sess.run(tf.constant("test tensorflow")))


if __name__ == '__main__':
    main()
