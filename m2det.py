import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

input_size = 800
batch_size = 32


def conv2d(input, filters, is_training):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=3, padding='same')
    return tf.nn.relu(tf.layers.batch_normalization(conv, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                                    training=is_training))


def vgg(input, is_training):
    conv1 = conv2d(input, 64, is_training)  # [N, 800, 800, 3]
    conv2 = conv2d(conv1, 64, is_training)
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='same')

    conv3 = conv2d(pool1, 128, is_training)
    conv4 = conv2d(conv3, 128, is_training)
    pool2 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2, padding='same')

    conv5 = conv2d(pool2, 256, is_training)
    conv6 = conv2d(conv5, 256, is_training)
    conv7 = conv2d(conv6, 256, is_training)
    pool3 = tf.layers.max_pooling2d(conv7, pool_size=2, strides=2, padding='same')

    conv8 = conv2d(pool3, 512, is_training)
    conv9 = conv2d(conv8, 512, is_training)
    conv10 = conv2d(conv9, 512, is_training)

    feature1 = conv10

    pool4 = tf.layers.max_pooling2d(conv10, pool_size=2, strides=2, padding='same')

    conv11 = conv2d(pool4, 512, is_training)
    conv12 = conv2d(conv11, 512, is_training)
    conv13 = conv2d(conv12, 512, is_training)
    pool5 = tf.layers.max_pooling2d(conv13, pool_size=2, strides=2, padding='same')

    conv14 = tf.layers.conv2d(pool5, filters=1024, kernel_size=3, strides=1, padding='same', dilation_rate=2)
    conv14 = tf.nn.relu(
        tf.layers.batch_normalization(conv14, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                      training=is_training))

    conv15 = conv2d(conv14, 1024, is_training)
    pool6 = tf.layers.max_pooling2d(conv15, pool_size=2, strides=2, padding='same')
    # Maybe, [N, 13, 13, 3]
    conv16 = tf.layers.conv2d(pool6, filters=1024, kernel_size=1, strides=1, padding='same')
    conv16 = tf.nn.relu(
        tf.layers.batch_normalization(conv16, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                      training=is_training))
    feature2 = conv16

    return feature1, feature2


def tum(x, scales, is_training):
    branch = [x]
    for i in range(scales - 1):
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same')
        x = tf.nn.relu(
            tf.layers.batch_normalization(x, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                          training=is_training))
        branch.insert(0, x)

    out = [x]
    for i in range(1, scales):
        x = tf.layers.conv2d(x, filters=256, kernel_size=3, strides=2, padding='same')
        x = tf.nn.relu(tf.layers.batch_normalization(x, axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                                     training=is_training))
        x = tf.image.resize_images(x, tf.shape(branch[i])[1:3], method=tf.image.ResizeMethod.BILINEAR)
        x = x + branch[i]
        out.append(x)

    for i in range(scales):
        out[i] = tf.layers.conv2d(out[i], filters=128, kernel_size=1, strides=1, padding='same')
        out[i] = tf.nn.relu(
            tf.layers.batch_normalization(out[i], axis=3, momentum=0.997, epsilon=1e-5, center=True, scale=True,
                                          training=is_training))

    return out


def sfam(input, planes, num_levels, num_scales, compress_ratio=16):
    print('wtf')


class M2Det:
    def __init__(self):
        self.is_training = tf.placeholder_with_default(False, shape=[None], name='is_training')
        self.num_classes = 2  # Kuzushiji or background.
        self.levels = 8
        self.scales = 6
        self.num_priors = 9

    def forward(self, input):
        with tf.variable_scope('VGG16'):
            feature1, feature2 = vgg(input, self.is_training)

        with tf.variable_scope('M2Det'):
            with tf.variable_scope('FFMv1'):
                feature1 = conv2d(feature1, 256, self.is_training)
                feature2 = tf.layers.conv2d(feature2, filters=512, kernel_size=1, strides=1, padding='same')
                feature2 = tf.nn.relu(
                    tf.layers.batch_normalization(feature2, axis=3, momentum=0.997, epsilon=1e-5, center=True,
                                                  scale=True,
                                                  training=self.is_training))
                feature2 = tf.image.resize_images(feature2, tf.shape(feature1)[1:3],
                                                  method=tf.image.ResizeMethod.BILINEAR)
                base_feature = tf.concat([feature1, feature2], axis=3)

            outs = []
            for i in range(self.levels):
                if i == 0:
                    net = tf.layers.conv2d(base_feature, filters=256, kernel_size=1, strides=1, padding='same')
                    net = tf.nn.relu(
                        tf.layers.batch_normalization(net, axis=3, momentum=0.997, epsilon=1e-5, center=True,
                                                      scale=True, training=self.is_training))
                else:
                    with tf.variable_scope('FFMv2_{}'.format(i + 1)):
                        net = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1)
                        net = tf.nn.relu(
                            tf.layers.batch_normalization(net, axis=3, momentum=0.997, epsilon=1e-5, center=True,
                                                          scale=True, training=self.is_training))
                        net = tf.concat([net, out[-1]], axis=3)
                with tf.variable_scope('TUM{}'.format(i + 1)):
                    out = tum(net, self.scales, self.is_training)

                outs.append(out)

            features = []

            for i in range(self.scales):
                feature = tf.concat([outs[j][i] for j in range(self.levels)], axis=3)

                with tf.variable_scope('SFAM'):
                    attention = tf.reduce_mean(feature, axis=[1, 2], keepdims=True)
                    attention = tf.layers.dense(attention, units=64, activation=tf.nn.relu, name='fc1_{}'.format(i + 1))
                    attention = tf.layers.dense(attention, units=1024, activation=tf.nn.sigmoid,
                                                name='fc2_{}'.format(i + 1))
                    feature = feature * attention

                features.insert(0, feature)

            all_cls = []
            all_reg = []
            with tf.variable_scope('prediction'):
                for i, feature in enumerate(features):
                    print(i + 1, feature.shape)
                    cls = tf.layers.conv2d(feature, filters=self.num_priors * self.num_classes, kernel_size=3,
                                           strides=1)
                    cls = tf.layers.batch_normalization(cls, axis=3, momentum=0.997, epsilon=1e-5, center=True,
                                                        scale=True, training=self.is_training)
                    sh = cls.shape
