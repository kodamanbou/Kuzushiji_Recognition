import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import math


def get_batch_data(line, prepro=True):
    filename = str(line[0].decode())
    image = cv2.cvtColor(cv2.imread(input_dir + 'train_images/' + filename + '.jpg'), cv2.COLOR_BGR2RGB)
    if prepro:
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
        image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)

    ratio_h = image.shape[0] / image_h
    ratio_w = image.shape[1] / image_w

    image = cv2.resize(image, (image_h, image_w))
    image = image / 255.

    cols = str(line[1].decode()).strip().split()
    boxes = []
    for i in range(len(cols) // 5):
        xmin = float(cols[i * 5 + 1]) / ratio_w
        ymin = float(cols[i * 5 + 2]) / ratio_h
        xmax = xmin + float(cols[i * 5 + 3]) / ratio_w
        ymax = ymin + float(cols[i * 5 + 4]) / ratio_h

        boxes.append([xmin, ymin, xmax, ymax])

    boxes = np.asarray(boxes, np.float32)

    y_true_60 = np.zeros((60, 36, 5), np.float32)
    y_true_30 = np.zeros((30, 18, 5), np.float32)
    y_true = [y_true_60, y_true_30]

    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
    box_sizes = np.expand_dims(box_sizes, 1)

    anchors = np.array([[36, 60], [18, 30]], np.float32)

    mins = np.maximum(-box_sizes / 2, -anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    whs = maxs - mins

    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
            box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] *
            whs[:, :, 1] + 1e-10)

    best_match_idx = np.argmax(iou, axis=1)
    for i, idx in enumerate(best_match_idx):
        ratio_x = image_w / anchors[idx, 0]
        ratio_y = image_h / anchors[idx, 1]
        x = int(np.floor(box_centers[i, 0] / ratio_x))
        y = int(np.floor(box_centers[i, 1] / ratio_y))

        y_true[idx][y, x, 0] = 1.
        y_true[idx][y, x, 1:3] = box_centers[i]
        y_true[idx][y, x, 3:5] = box_sizes[i]

    return image, y_true_60, y_true_30


def conv2d(input, filters, kernel_size, strides):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                            activation=tf.nn.relu)
    return conv


def ocr_network(input, is_training):
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
    dense7 = tf.layers.dense(dense4, units=1)
    dense8 = tf.layers.dense(dense5, units=2, activation=tf.nn.sigmoid)
    dense9 = tf.layers.dense(dense6, units=2, activation=tf.nn.sigmoid)

    concat1 = tf.concat([dense7, dense8, dense9], axis=3)  # [N, 60, 36, 5]

    dense10 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense11 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense12 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)

    dense13 = tf.layers.dense(dense10, units=64, activation=tf.nn.relu)
    dense14 = tf.layers.dense(dense11, units=64, activation=tf.nn.relu)
    dense15 = tf.layers.dense(dense12, units=64, activation=tf.nn.relu)

    dense16 = tf.layers.dense(dense13, units=1)
    dense17 = tf.layers.dense(dense14, units=2, activation=tf.nn.sigmoid)
    dense18 = tf.layers.dense(dense15, units=2, activation=tf.nn.sigmoid)

    concat2 = tf.concat([dense16, dense17, dense18], axis=3)  # [N, 30, 18, 5]

    return concat1, concat2


def predict(feature_maps):
    results = [reorg(feature_map) for feature_map in feature_maps]

    def _reshape(result):
        xy_offset, boxes, conf_logits = result
        grid_size = tf.shape(xy_offset)[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1], 4])
        conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1], 1])
        return boxes, conf_logits

    boxes_list, confs_list = [], []
    for result in results:
        boxes, conf_logits = _reshape(result)
        confs = tf.nn.sigmoid(conf_logits)
        boxes_list.append(boxes)
        confs_list.append(confs)

    boxes = tf.concat(boxes_list, axis=1)
    confs = tf.concat(confs_list, axis=1)

    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    xmin = center_x - width / 2
    xmax = center_x + width / 2
    ymin = center_y - height / 2
    ymax = center_y + height / 2

    boxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    return boxes, confs


def reorg(feature_map):
    grid_size = tf.shape(feature_map)[1:3]
    image_size = tf.cast([image_h, image_w], tf.int32)
    ratio = tf.cast(image_size / grid_size, tf.float32)

    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    xy_offset = tf.concat([x_offset, y_offset], axis=-1)
    xy_offset = tf.cast(tf.reshape(xy_offset, shape=[grid_size[0], grid_size[1], 1, 2]), tf.float32)

    box_centers = feature_map[..., 1:3] + xy_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = feature_map[..., 3:5]
    box_sizes = tf.exp(box_sizes) * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)
    conf_logits = feature_map[..., 0]

    return xy_offset, boxes, conf_logits


def compute_loss(y_pred, y_true):
    # Compute loss each layer.
    N = tf.cast(tf.shape(y_pred)[0], tf.float32)
    grid_size = tf.shape(y_pred)[1:3]
    image_size = tf.cast([image_h, image_w], tf.int32)
    ratio = tf.cast(image_size / grid_size, tf.float32)

    xy_offset, pred_boxes, conf_logits = reorg(y_pred)

    object_mask = y_true[..., 4:5]  # [N, 60, 36, 1]
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_cond(idx, ignore_mask):
        return tf.less(idx, tf.cast(N, tf.int32))

    def loop_body(idx, ignore_mask):
        valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
        iou = box_iou(pred_boxes[idx], valid_true_boxes)
        best_iou = tf.reduce_max(iou, axis=-1)  # [60, 36, 1]

        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx + 1, ignore_mask

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)  # [N, 60, 36, 1]

    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    true_xy = y_true[..., 0:2] / ratio[::-1] - xy_offset
    pred_xy = pred_box_xy / ratio[::-1] - xy_offset
    true_tw_th = y_true[..., 2:4] / ratio[::-1]
    pred_tw_th = pred_box_wh / ratio[::-1]

    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                          x=tf.ones_like(true_tw_th), y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                          x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
    pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask) / N
    wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask) / N

    conf_pos_mask = object_mask
    conf_neg_mask = (1 - object_mask) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conf_logits)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conf_logits)
    conf_loss = conf_loss_pos + conf_loss_neg

    # focal loss
    alpha = 1.0
    gamma = 2.0
    focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(conf_logits)), gamma)
    conf_loss *= focal_mask

    conf_loss = tf.reduce_sum(conf_loss) / N

    return xy_loss, wh_loss, conf_loss


def box_iou(pred_boxes, valid_true_boxes):
    pred_boxes_xy = pred_boxes[..., 0:2]
    pred_boxes_wh = pred_boxes[..., 2:4]
    pred_boxes_xy = tf.expand_dims(pred_boxes_xy, -2)  # [60, 36, 1, 2]
    pred_boxes_wh = tf.expand_dims(pred_boxes_wh, -2)

    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    intersect_mins = tf.maximum(pred_boxes_xy - pred_boxes_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_boxes_xy + pred_boxes_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # [60, 36, V]
    pred_box_area = pred_boxes_wh[..., 0] * pred_boxes_wh[..., 1]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)  # [60, 36, V]

    return iou


if __name__ == '__main__':
    image_h = 728
    image_w = 448
    batch_size = 3
    input_dir = '../input/'

    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
    df_train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    df_train.dropna(inplace=True)
    df_train.reset_index(inplace=True, drop=True)

    train_batch_num = int(math.ceil(float(len(df_train)) / batch_size))
    train_dataset = tf.data.Dataset.from_tensor_slices(df_train.values)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x],
                             Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=16
    )
    train_dataset = train_dataset.shuffle(len(df_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    image, y_true_60, y_true_30 = iterator.get_next()
    image.set_shape([None, None, None, 3])
    y_true_60.set_shape([None, None, None])
    y_true_30.set_shape([None, None, None])

    y_pred_60, y_pred_30 = ocr_network(image, is_training=is_training)
    loss_xy_60, loss_wh_60, loss_conf_60 = compute_loss(y_pred_60, y_true_60)
    loss_xy_30, loss_wh_30, loss_conf_30 = compute_loss(y_pred_30, y_true_30)

    total_loss = loss_xy_60 + loss_wh_60 + loss_conf_60 + loss_xy_30 + loss_wh_30 + loss_conf_30

    pred_boxes, pred_confs = predict([y_pred_60, y_pred_30])

    global_step = tf.Variable(0, trainable=False, name='global_step')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = opt.compute_gradients(total_loss)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training section.
    print('Training start.')
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for epoch in range(15):
            sess.run(train_init_op)
            for i in range(train_batch_num):
                _, _total_loss, _pred_boxes, _global_step = sess.run([train_op, total_loss, pred_boxes, global_step])

                if _global_step % 1000 == 0:
                    print('global step: {}\ttotal loss: {}\tbox_num: {}'.format(_global_step, _total_loss,
                                                                                len(_pred_boxes)))
                    saver.save(sess, 'ocr_model.ckpt')
