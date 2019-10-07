import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import math

input_size = 800
batch_size = 16
input_dir = '../input/'


def get_batch_data(line, priors, prepro=False):
    filename = str(line[0].decode())
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if prepro:
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
        image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)

    h_ratio = image.shape[0] / 800.
    w_ratio = image.shape[1] / 800.
    image = cv2.resize(image, (800, 800))
    image = np.asarray(image, np.float32)
    cols = str(line[1].decode()).strip().split()

    boxes = []

    for i in range(len(cols) // 5):
        label = 0
        xmin = float(cols[i * 5 + 1]) / w_ratio
        ymin = float(cols[i * 5 + 2]) / h_ratio
        xmax = xmin + float(cols[i * 5 + 3]) / w_ratio
        ymax = ymin + float(cols[i * 5 * 4]) / h_ratio

        boxes.append([xmin, ymin, xmax, ymax, label])

    boxes = np.asarray(boxes, np.float32)
    assignment = assign_boxes(boxes, priors, num_classes=1, threshold=0.45)

    return image, assignment


def assign_boxes(boxes, priors, num_classes, threshold=0.5):
    num_classes += 1  # add background class.
    assignment = np.zeros((len(priors), 4 + num_classes + 1))
    assignment[:, 4] = 1.0
    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes, priors, threshold)
    encoded_boxes = encoded_boxes.reshape(-1, len(priors), 5)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
    best_iou_mask = best_iou > 0
    best_iou_idx = best_iou_idx[best_iou_mask]
    assign_num = len(best_iou_idx)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
    assignment[:, 4][best_iou_mask] = 0
    assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
    assignment[:, :-1][best_iou_mask] = 1  # objectness

    return assignment


def encode_box(box, priors, assignment_threshold=0.45):
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:], box[2:])
    inter_wh = np.maximum(inter_botright - inter_upleft, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    iou = inter / union

    encoded_box = np.zeros((len(priors), 5))
    assign_mask = iou >= assignment_threshold
    encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = priors[assign_mask]
    box_center = 0.5 * (box[2:] - box[:2])
    box_wh = box[:2] - box[2:]
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:])
    assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]

    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] /= 0.1
    encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
    encoded_box[:, 2:4][assign_mask] /= 0.2

    return encoded_box.ravel()


def generate_priors(num_scales=3, anchor_scale=2.0, image_size=320, shapes=(40, 20, 10, 5, 3, 1)):
    anchor_configs = {}
    for shape in shapes:
        anchor_configs[shape] = []
        for scale_octave in range(num_scales):
            for aspect_ratio in [(1, 1), (1.41, 0.71), (0.71, 1.41)]:
                anchor_configs[shape].append(
                    (image_size / shape, scale_octave / float(num_scales), aspect_ratio))

    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            base_anchor_size = anchor_scale * stride * (2 ** octave_scale)
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0
            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_level /= image_size
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)

    return anchor_boxes


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


def calc_focal_loss(cls_outputs, cls_targets, alpha=0.25, gamma=2.0):
    positive_mask = tf.equal(cls_targets, 1.0)
    pos = tf.where(positive_mask, 1.0 - cls_outputs, tf.zeros_like(cls_outputs))
    neg = tf.where(positive_mask, tf.zeros_like(cls_outputs), cls_outputs)
    pos_loss = - alpha * tf.pow(pos, gamma) * tf.log(tf.clip_by_value(cls_outputs, 1e-15, 1.0))
    neg_loss = - (1 - alpha) * tf.pow(neg, gamma) * tf.log(tf.clip_by_value(cls_outputs, 1e-15, 1.0))
    loss = tf.reduce_sum(pos_loss + neg_loss, axis=[1, 2])
    return loss


def calc_cls_loss(cls_outputs, cls_targets, positive_flag):
    batch_size = tf.shape(cls_outputs)[0]
    num_anchors = tf.to_float(tf.shape(cls_outputs)[1])
    num_positives = tf.reduce_sum(positive_flag, axis=-1)
    num_negatives = tf.minimum(3 * num_positives, num_anchors - num_positives)
    negative_mask = tf.greater(num_negatives, 0)

    cls_outputs = tf.clip_by_value(cls_outputs, 1e-15, 1 - 1e-15)
    conf_loss = - tf.reduce_sum(cls_targets * tf.log(cls_outputs), axis=-1)
    pos_conf_loss = tf.reduce_sum(conf_loss * positive_flag, axis=1)

    has_min = tf.to_float(tf.reduce_any(negative_mask))
    num_neg = tf.concat(axis=0, values=[num_negatives, [(1 - has_min) * 100]])

    num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives, tf.greater(num_negatives, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    max_confs = tf.reduce_max(cls_outputs[:, :, 1:], axis=2)
    _, indices = tf.nn.top_k(max_confs * (1 - positive_flag), k=num_neg_batch)
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_anchors) + tf.reshape(indices, [-1]))
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

    cls_loss = pos_conf_loss + neg_conf_loss
    cls_loss /= (num_positives + tf.to_float(num_neg_batch))

    return cls_loss


def calc_box_loss(box_outputs, box_targets, positive_flag, delta=0.1):
    num_positives = tf.reduce_sum(positive_flag, axis=-1)
    normalizer = num_positives * 4
    normalizer = tf.where(tf.not_equal(normalizer, 0), normalizer, tf.ones_like(normalizer))

    loss_scale = 2.0 - box_targets[:, :, 2:3] * box_targets[:, :, 3:4]

    sq_loss = 0.5 * (box_targets - box_outputs) ** 2
    abs_loss = 0.5 * delta ** 2 + delta * (tf.abs(box_outputs - box_targets) - delta)
    l1_loss = tf.where(tf.less(tf.abs(box_outputs - box_targets), delta), sq_loss, abs_loss)

    box_loss = tf.reduce_sum(l1_loss, axis=-1, keepdims=True)
    box_loss = box_loss * loss_scale
    box_loss = tf.reduce_sum(box_loss, axis=-1)
    box_loss = tf.reduce_sum(box_loss * positive_flag, axis=-1)
    box_loss = box_loss / normalizer

    return box_loss


def calc_loss(y_true, y_pred, box_loss_weight):
    box_outputs = y_pred[:, :, :4]
    box_targets = y_true[:, :, :4]
    cls_outputs = y_pred[:, :, 4:]
    cls_targets = y_true[:, :, 4:-1]
    positive_flag = y_true[:, :, -1]
    num_positives = tf.reduce_sum(positive_flag, axis=-1)

    box_loss = calc_box_loss(box_outputs, box_targets, positive_flag)
    cls_loss = calc_focal_loss(cls_outputs, cls_targets)

    total_loss = cls_loss + box_loss_weight * box_loss

    return tf.reduce_mean(total_loss)


class M2Det:
    def __init__(self, inputs, is_training):
        self.is_training = is_training
        self.num_classes = 1  # Kuzushiji or not.
        self.levels = 8
        self.scales = 6
        self.num_priors = 9
        self.inputs = inputs

    def build(self):
        with tf.variable_scope('VGG16'):
            feature1, feature2 = vgg(self.inputs, self.is_training)

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
                    cls = tf.reshape(cls, shape=[-1, sh[1], sh[2], sh[3]])
                    all_cls.append(cls)

                    reg = tf.layers.conv2d(feature, filters=self.num_priors * 4, kernel_size=3, strides=1)
                    reg = tf.layers.batch_normalization(reg, axis=2, momentum=0.997, epsilon=1e-5, scale=True,
                                                        center=True, training=self.is_training)
                    sh = reg.shape
                    reg = tf.reshape(reg, shape=[-1, sh[1], sh[2], sh[3]])
                    all_reg.append(reg)

                    all_cls = tf.concat(all_cls, axis=1)
                    all_reg = tf.concat(all_reg, axis=1)
                    num_boxes = int(all_reg.shape[-1].value / 4)
                    all_cls = tf.reshape(all_cls, shape=[-1, num_boxes, self.num_classes])
                    all_cls = tf.nn.softmax(all_cls)
                    all_reg = tf.reshape(all_reg, shape=[-1, num_boxes, 4])
                    self.prediction = tf.concat([all_reg, all_cls], axis=-1)


if __name__ == '__main__':
    is_training = tf.placeholder_with_default(False, shape=[None], name='is_training')

    df_train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    df_train.dropna(inplace=True)
    df_train.reset_index(inplace=True, drop=True)

    train_batch_num = int(math.ceil(float(len(df_train)) / batch_size))
    train_dataset = tf.data.Dataset.from_tensor_slices(df_train.values)
    priors = generate_priors(image_size=input_size, shapes=[100, 50, 25, 13, 8, 3])
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, priors, True],
                             Tout=[tf.float32, tf.float32]),
        num_parallel_calls=16
    )
    train_dataset = train_dataset.shuffle(len(df_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    image, y_true = iterator.get_next()
    image.set_shape([None, None, None, 3])
    y_true.set_shape([None, None, None])

    net = M2Det(image, is_training)
    y_pred = net.prediction
    total_loss = calc_loss(y_true, y_pred, box_loss_weight=20.0)

    weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'kernel' in v.name]
    decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights])) * 1e-3
    total_loss += decay

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_var = tf.trainable_variables
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9)
        grads = tf.gradients(total_loss, train_var)
        train_op = opt.apply_gradients(zip(grads, train_var), global_step=global_step)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training section.
    print('Training start.')

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        for epoch in range(150):
            sess.run(train_init_op)

            for i in range(train_batch_num):
                _, _y_pred, loss_value, _global_step = sess.run([train_op, y_pred, total_loss, global_step],
                                                                feed_dict={is_training: True})

                if _global_step % 1000 == 0:
                    print('Global Step: {} \ttotal_loss: {}'.format(_global_step, loss_value))
