import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_batch_data(line, prepro=False):
    print('wtf')


def conv2d(input, filters, kernel_size, strides):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            activation=tf.nn.relu)
    return conv


def vgg16(input, is_training):
    conv1 = conv2d(input, 64, 3, 1)  # [N, 728, 448, 64]
    conv2 = conv2d(conv1, 64, 3, 1)
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)  # [N, 364, 224, 64]

    conv3 = conv2d(pool1, 64, 3, 1)  # [N, 364, 224, 64]
    dropout1 = tf.layers.dropout(pool1, rate=0.25, training=is_training)

    conv4 = conv2d(conv3, 64, 3, 1)
    conv5 = conv2d(dropout1, 128, 8, 6)  # [N, 60, 36, 128]

    pool2 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)  # [N, 182, 112, 64]
    dropout2 = tf.layers.dropout(conv5, rate=0.5, training=is_training)  # [N, 60, 36, 128]

    dropout3 = tf.layers.dropout(pool2, rate=0.25, training=is_training)
    dense1 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)

    conv6 = conv2d(dropout3, 128, 5, 6)  # [N, 30, 18, 128]
    dense4 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu)
    dense5 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu)
    dense6 = tf.layers.dense(dense3, units=64, activation=tf.nn.relu)

    dropout4 = tf.layers.dropout(conv6, rate=0.5, training=is_training)
    dense7 = tf.layers.dense(dense4, units=1, activation=tf.nn.sigmoid)
    dense8 = tf.layers.dense(dense5, units=2, activation=tf.nn.tanh)
    dense9 = tf.layers.dense(dense6, units=2, activation=tf.nn.sigmoid)

    concat1 = tf.concat([dense7, dense8, dense9], axis=3)  # [N, 60, 36, 5]

    dense10 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense11 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense12 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)

    dense13 = tf.layers.dense(dense10, units=64, activation=tf.nn.relu)
    dense14 = tf.layers.dense(dense11, units=64, activation=tf.nn.relu)
    dense15 = tf.layers.dense(dense12, units=64, activation=tf.nn.relu)

    dense16 = tf.layers.dense(dense13, units=1, activation=tf.nn.sigmoid)
    dense17 = tf.layers.dense(dense14, units=2, activation=tf.nn.tanh)
    dense18 = tf.layers.dense(dense15, units=2, activation=tf.nn.sigmoid)

    concat2 = tf.concat([dense16, dense17, dense18], axis=3)  # [N, 30, 18, 5]


if __name__ == '__main__':
    image_h = 728
    image_w = 448
    num_classes = 4212
    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
