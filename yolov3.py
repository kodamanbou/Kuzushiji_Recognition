import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

df_train = pd.read_csv('../input/train.csv')
df_train.dropna(inplace=True)
df_train.reset_index(inplace=True, drop=True)
unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}


def conv2d(inputs, filters, kernel_size, strides=1, is_training=False):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)

    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                              padding=('same' if strides == 1 else 'valid'))
    inputs = tf.nn.leaky_relu(tf.layers.batch_normalization(inputs, training=is_training))

    return inputs


def darknet53_body(inputs, is_training=False):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters, 1, is_training=is_training)
        net = conv2d(net, filters * 2, 3, is_training=is_training)

        net = net + shortcut

        return net

    net = conv2d(inputs, 32, 3, strides=1, is_training=is_training)
    net = conv2d(net, 64, 3, strides=2, is_training=is_training)

    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2, is_training=is_training)

    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2, is_training=is_training)

    for i in range(8):
        net = res_block(net, 128)


class Graph:
    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, use_static_shape=True):
        self.class_num = class_num
        self.anchors = anchors
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.use_static_shape = use_static_shape

        self.x = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name='X')
        self.is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
