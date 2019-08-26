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

    def forward(self, inputs, is_training=False):
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

        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        box_sizes = box_sizes * ratio[::-1]

        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        return xy_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        feature_map1, feature_map2, feature_map3 = feature_maps

        feature_map_anchors = [(feature_map1, self.anchors[6:9]),
                               (feature_map2, self.anchors[3:6]),
                               (feature_map3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for feature_map, anchors in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])

            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.nn.sigmoid(conf_logits)
            probs = tf.nn.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def loss_layer(self, feature_map, y_true, anchors):
        grid_size = tf.shape(feature_map)[1:3]
        ratio = tf.cast(self.image_size / grid_size, tf.float32)
        N = tf.cast(tf.shape(feature_map)[0], tf.float32)

        xy_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map, anchors)

        # get mask.
        object_mask = y_true[..., 4:5]
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        def loop_body(idx, ignore_mask):
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            best_iou = tf.reduce_max(iou, axis=-1)

            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask

        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        true_xy = y_true[..., 0:2] / ratio[::-1] - xy_offset
        pred_xy = pred_box_xy / ratio[::-1] - xy_offset

        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors

        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
