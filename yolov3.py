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

    regularizer = tf.contrib.layers.l2_regularizer(5e-4)
    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                              padding=('same' if strides == 1 else 'valid'),
                              kernel_regularizer=regularizer)
    inputs = tf.nn.leaky_relu(tf.layers.batch_normalization(inputs, epsilon=1e-05, training=is_training))

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

    route1 = net
    net = conv2d(net, 512, 3, strides=2, is_training=is_training)

    for i in range(8):
        net = res_block(net, 256)

    route2 = net
    net = conv2d(net, 1024, 3, strides=2, is_training=is_training)

    for i in range(4):
        net = res_block(net, 512)

    route3 = net

    return route1, route2, route3


def yolo_block(inputs, filters, is_training=False):
    net = conv2d(inputs, filters, 1, is_training=is_training)
    net = conv2d(net, filters * 2, 3, is_training=is_training)
    net = conv2d(net, filters, 1, is_training=is_training)
    net = conv2d(net, filters * 2, 3, is_training=is_training)
    net = conv2d(net, filters, 1, is_training=is_training)
    route = net
    net = conv2d(net, filters * 2, 3, is_training=is_training)

    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs


class Graph:
    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False):
        self.class_num = class_num
        self.anchors = anchors
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss

    def forward(self, inputs, is_training=False, reuse=False):
        self.image_size = tf.shape(inputs)[1:3]

        with tf.variable_scope('darknet53_body'):
            route1, route2, route3 = darknet53_body(inputs, is_training=is_training)

        with tf.variable_scope('yolov3_head'):
            inter1, net = yolo_block(route3, 512, is_training=is_training)
            feature_map1 = tf.layers.conv2d(net, 3 * (5 + self.class_num), 1, strides=1)
            feature_map1 = tf.identity(feature_map1, name='feature_map_1')

            inter1 = conv2d(inter1, 256, 1, is_training=is_training)
            inter1 = upsample_layer(inter1, route2.get_shape().as_list())
            concat1 = tf.concat([inter1, route2], axis=3)

            inter2, net = yolo_block(concat1, 256, is_training=is_training)
            feature_map2 = tf.layers.conv2d(net, 3 * (5 + self.class_num), 1, strides=1)
            feature_map2 = tf.identity(feature_map2, name='feature_map_2')

            inter2 = conv2d(inter2, 128, 1, is_training=is_training)
            inter2 = upsample_layer(inter2, route1.get_shape().as_list())
            concat2 = tf.concat([inter2, route1], axis=3)

            _, feature_map3 = yolo_block(concat2, 128, is_training=is_training)
            feature_map3 = tf.layers.conv2d(feature_map3, 3 * (5 + self.class_num), 1, strides=1)
            feature_map3 = tf.identity(feature_map3, name='feature_map_3')

        return feature_map1, feature_map2, feature_map3

    def reorg_layer(self, feature_map, anchors):
        grid_size = feature_map.get_shape().as_list()[1:3]
        ratio = tf.cast(self.image_size / grid_size, tf.float32)
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        xy_offset = tf.concat([x_offset, y_offset], axis=-1)
        xy_offset = tf.cast(tf.reshape(xy_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        box_centers = box_centers + xy_offset
        box_centers = box_centers * ratio[::-1]
