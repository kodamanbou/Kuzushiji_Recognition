import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import pandas as pd
import tensorflow as tf

df_translation = pd.read_csv('../input/unicode_translation.csv')
fontsize = 50
font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', size=fontsize, encoding='utf-8')


def unicode_to_num(unicode):
    return df_translation[df_translation['Unicode'] == unicode].index[0]


def visualize(image_fn, boxes, scores, labels):
    imsource = Image.open(image_fn).convert('RGBA').resize((416, 416))
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        bbox_draw.rectangle((xmin, ymin, xmax, ymax), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        char_draw.text((xmax + fontsize / 4, ymax / 2 - fontsize), str(labels[i]), fill=(0, 0, 255, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert('RGB')
    return np.asarray(imsource)


def get_batch_data(line, class_num, anchors):
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    filename = str(line[0].decode())
    image = Image.open('../input/train_images/' + filename + '.jpg')
    w_ratio = image.width / 416.
    h_ratio = image.height / 416.
    image = image.resize((416, 416))
    image = np.asarray(image, np.float32)
    cols = str(line[1].decode()).strip().split(' ')

    labels = []
    boxes = []

    for i in range(len(cols) // 5):
        label = int(unicode_to_num(cols[i * 5]))
        xmin = float(cols[i * 5 + 1]) / w_ratio
        ymin = float(cols[i * 5 + 2]) / h_ratio
        xmax = xmin + float(cols[i * 5 + 3]) / w_ratio
        ymax = ymin + float(cols[i * 5 + 4]) / h_ratio

        labels.append(label)
        boxes.append([xmin, ymin, xmax, ymax])

    labels = np.asarray(labels, np.int64)
    boxes = np.asarray(boxes, np.float32)
    boxes = np.concatenate((boxes, np.full(shape=(boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)

    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]

    y_true_13 = np.zeros((13, 13, 3, 6 + class_num), np.float32)
    y_true_26 = np.zeros((26, 26, 3, 6 + class_num), np.float32)
    y_true_52 = np.zeros((52, 52, 3, 6 + class_num), np.float32)

    y_true_13[..., -1] = 1.
    y_true_26[..., -1] = 1.
    y_true_52[..., -1] = 1.

    y_true = [y_true_13, y_true_26, y_true_52]

    box_sizes = np.expand_dims(box_sizes, 1)
    mins = np.maximum(-box_sizes / 2, -anchors / 2)
    maxs = np.minimum(box_sizes / 2, anchors / 2)
    # [N, 9, 2]
    whs = maxs - mins

    iou = (whs[:, :, 0] * whs[:, :, 1]) / (
            box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :, 1]
            + 1e-10)

    best_match_index = np.argmax(iou, axis=1)
    ratio_dict = {1.: 8., 2.: 16., 3.: 32}
    for i, idx in enumerate(best_match_index):
        feature_map_group = 2 - idx // 3
        ratio = ratio_dict[np.ceil((idx + 1) / 3.)]
        x = int(np.floor(box_centers[i, 0] / ratio))
        y = int(np.floor(box_centers[i, 1] / ratio))
        k = anchors_mask[feature_map_group].index(idx)
        c = labels[i]

        y_true[feature_map_group][y, x, k, :2] = box_centers[i]
        y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
        y_true[feature_map_group][y, x, k, 4] = 1.
        y_true[feature_map_group][y, x, k, 5 + c] = 1.
        y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]

    return image, y_true_13, y_true_26, y_true_52


def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Perform NMS on GPU using TensorFlow.
    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4])  # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:, i])
        filter_score = tf.boolean_mask(score[:, i], mask[:, i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


# Network utils.
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


# YOLO-v3.
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
            inter1 = upsample_layer(inter1, tf.shape(route2))
            concat1 = tf.concat([inter1, route2], axis=3)

            inter2, net = yolo_block(concat1, 256, is_training=is_training)
            feature_map2 = tf.layers.conv2d(net, 3 * (5 + self.class_num), 1, strides=1)
            feature_map2 = tf.identity(feature_map2, name='feature_map_2')

            inter2 = conv2d(inter2, 128, 1, is_training=is_training)
            inter2 = upsample_layer(inter2, tf.shape(route1))
            concat2 = tf.concat([inter2, route1], axis=3)

            _, feature_map3 = yolo_block(concat2, 128, is_training=is_training)
            feature_map3 = tf.layers.conv2d(feature_map3, 3 * (5 + self.class_num), 1, strides=1)
            feature_map3 = tf.identity(feature_map3, name='feature_map_3')

        return feature_map1, feature_map2, feature_map3

    def reorg_layer(self, feature_map, anchors):
        grid_size = tf.shape(feature_map)[1:3]
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
            grid_size = tf.shape(x_y_offset)[:2]
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

        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.image_size[1], tf.float32)) * (
                y_true[..., 3:4] / tf.cast(self.image_size[0], tf.float32))

        # Loss part.
        mix_w = y_true[..., -1:]
        xy_loss = tf.reduce_mean(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_mean(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            focal_loss = alpha * tf.pow(tf.abs(object_mask - tf.nn.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_loss

        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def box_iou(self, pred_boxes, valid_true_boxes):
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0)

        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    def compute_loss(self, y_pred, y_true):
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9],
                        self.anchors[3:6],
                        self.anchors[0:3]]

        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class

        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]


if __name__ == '__main__':
    batch_size = 6
    class_num = 4787
    anchors = [[10, 13], [16, 30], [33, 23],
               [30, 61], [62, 45], [59, 119],
               [116, 90], [156, 198], [373, 326]]

    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')

    # data pipeline.
    df_train = pd.read_csv('../input/train.csv')
    df_train.dropna(inplace=True)
    df_train.reset_index(inplace=True, drop=True)

    train_batch_num = int(math.ceil(float(len(df_train)) / batch_size))

    train_dataset = tf.data.Dataset.from_tensor_slices(df_train.values)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, class_num, anchors],
                             Tout=[tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=16
    )
    train_dataset = train_dataset.shuffle(len(df_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    val_data = tf.placeholder(tf.float32, shape=[1, 416, 416, 3], name='X')  # For debug.

    yolo_model = Graph(class_num, anchors)
    with tf.variable_scope('yolov3', reuse=tf.AUTO_REUSE):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)
        val_feature_maps = yolo_model.forward(val_data, False)

    loss = yolo_model.compute_loss(pred_feature_maps, y_true)
    y_pred = yolo_model.predict(pred_feature_maps)

    l2_loss = tf.losses.get_regularization_loss()

    global_step = tf.Variable(0, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    learning_rate = tf.train.piecewise_constant(global_step, boundaries=[30, 50], values=[1e-4, 3e-5, 1e-5],
                                                name='piecewise_learning_rate')

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_vars = tf.contrib.framework.get_variables_to_restore(include=['yolov3/yolov3_head'])
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
        grads_and_vars = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]]
                          for gv in gvs]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        print('Training start.')
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch in range(100):
            sess.run(train_init_op)

            for i in range(train_batch_num):
                _, _y_pred, _y_true, _loss, _global_step, _lr = sess.run(
                    [train_op, y_pred, y_true, loss, global_step, learning_rate],
                    feed_dict={is_training: True}
                )

                if _global_step % 1000 == 0 and _global_step > 0:
                    print('Epoch: {} \tloss: total: {} \txy: {} \twh: {} \tconf: {} \tclass{}'
                          .format(_global_step, _loss[0], _loss[1], _loss[2], _loss[3], _loss[4]))

                    test_img = Image.open('../input/test_images/test_0a9b81ce.jpg')
                    test_h, test_w = test_img.height, test_img.width
                    test_img = np.asarray(test_img.resize((416, 416)))

                    pred_boxes, pred_confs, pred_probs = yolo_model.predict(val_feature_maps)
                    pred_scores = pred_confs * pred_probs

                    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, class_num, max_boxes=200)
                    _boxes, _scores, _labels = sess.run([boxes, scores, labels], feed_dict={val_data: test_img})

                    plt.figure(figsize=(15, 15))
                    plt.imshow(visualize('../input/test_images/test_0a9b81ce.jpg', _boxes, _scores, _labels))
                    plt.show()

    print('Training end.')
