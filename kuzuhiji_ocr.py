import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import math


def create_dataset():
    if os.path.exists('dataset'):
        return
    else:
        os.mkdir('dataset')

    for f in os.listdir(os.path.join(input_dir, 'train_images')):
        image = cv2.cvtColor(cv2.imread(input_dir + 'train_images/' + f), cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
        image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
        cv2.imwrite(os.path.join('dataset', f), image)
        print(f, ' done.')

    os.makedirs('dataset/validate')
    for f in os.listdir(os.path.join(input_dir, 'test_images')):
        image = cv2.cvtColor(cv2.imread(input_dir + 'test_images/' + f), cv2.COLOR_BGR2RGB)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
        image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)
        cv2.imwrite(os.path.join('dataset/validate', f), image)
        print(f, ' done.')


def visualize(image_fn, boxes, scores, fontsize=20):
    font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', size=fontsize, encoding='utf-8')
    imsource = Image.open(image_fn).convert('RGBA').resize((image_w, image_h))
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        bbox_draw.rectangle((xmin, ymin, xmax, ymax), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        char_draw.text((xmax + fontsize / 4, ymax / 2 - fontsize), str(scores[i]), fill=(0, 0, 255, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert('RGB')
    return np.asarray(imsource)


def gpu_nms(boxes, scores, num_classes, max_boxes=1200, score_thresh=0.5, nms_thresh=0.5):
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
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)

    return boxes, score


def get_batch_data(line, prepro=False):
    filename = str(line[0].decode())
    if prepro:
        image = cv2.cvtColor(cv2.imread('dataset/' + filename + '.jpg'), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(cv2.imread(input_dir + 'train_images/' + filename + '.jpg'), cv2.COLOR_BGR2RGB)

    ratio_h = image.shape[0] / image_h
    ratio_w = image.shape[1] / image_w

    image = cv2.resize(image, (image_w, image_h))
    image = np.asarray(image, np.float32)
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

    y_true_l = np.zeros((26, 16, 5), np.float32)
    y_true_s = np.zeros((13, 8, 5), np.float32)
    y_true = [y_true_l, y_true_s]

    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
    box_sizes = np.expand_dims(box_sizes, 1)

    anchors = np.array([[16, 26], [8, 13]], np.float32)

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

        y_true[idx][y, x, 0:2] = box_centers[i]
        y_true[idx][y, x, 2:4] = box_sizes[i]
        y_true[idx][y, x, 4] = 1.

    return image, y_true_l, y_true_s


def get_val_data(filename, prepro=False):
    filename = str(filename.decode())
    if prepro:
        image = cv2.cvtColor(cv2.imread('dataset/validate/' + filename), cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(cv2.imread(input_dir + 'test_images/' + filename), cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (image_w, image_h))
    image = np.asarray(image, np.float32)
    image = image / 255.

    image_id = filename.split('.')[0]
    image_id = np.char.encode(image_id, encoding='utf-8')

    return image, image_id


def conv2d(input, filters, kernel_size, strides, padding='same'):
    conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            activation=tf.nn.relu)
    return conv


def ocr_network(input, is_training):
    conv1 = conv2d(input, 64, 3, 1)  # [N, 728, 448, 64]
    conv2 = conv2d(conv1, 64, 3, 1)
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)  # [N, 364, 224, 64]

    conv3 = conv2d(pool1, 64, 3, 1)  # [N, 364, 224, 64]
    dropout1 = tf.layers.dropout(pool1, rate=0.25, training=is_training)

    conv4 = conv2d(conv3, 64, 3, 1)
    conv5 = conv2d(dropout1, 128, 12, 14)  # [N, 26, 16, 128]

    pool2 = tf.layers.max_pooling2d(conv4, pool_size=2, strides=2)  # [N, 182, 112, 64]
    dropout2 = tf.layers.dropout(conv5, rate=0.5, training=is_training)  # [N, 26, 16, 128]

    dropout3 = tf.layers.dropout(pool2, rate=0.25, training=is_training)
    dense1 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dropout2, units=128, activation=tf.nn.relu)

    conv6 = conv2d(dropout3, 128, 12, 14)  # [N, 13, 8, 128]
    dense4 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu)
    dense5 = tf.layers.dense(dense2, units=64, activation=tf.nn.relu)
    dense6 = tf.layers.dense(dense3, units=64, activation=tf.nn.relu)

    dropout4 = tf.layers.dropout(conv6, rate=0.5, training=is_training)
    dense7 = tf.layers.dense(dense4, units=1)
    dense8 = tf.layers.dense(dense5, units=2, activation=tf.nn.sigmoid)
    dense9 = tf.layers.dense(dense6, units=2, activation=tf.nn.sigmoid)

    concat1 = tf.concat([dense7, dense8, dense9], axis=-1)  # [N, 26, 16, 5]

    dense10 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense11 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)
    dense12 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu)

    dense13 = tf.layers.dense(dense10, units=64, activation=tf.nn.relu)
    dense14 = tf.layers.dense(dense11, units=64, activation=tf.nn.relu)
    dense15 = tf.layers.dense(dense12, units=64, activation=tf.nn.relu)

    dense16 = tf.layers.dense(dense13, units=1)
    dense17 = tf.layers.dense(dense14, units=2, activation=tf.nn.sigmoid)
    dense18 = tf.layers.dense(dense15, units=2, activation=tf.nn.sigmoid)

    concat2 = tf.concat([dense16, dense17, dense18], axis=-1)  # [N, 13, 8, 5]

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
    xy_offset = tf.cast(tf.reshape(xy_offset, shape=[grid_size[0], grid_size[1], 2]), tf.float32)

    conf_logits, box_centers, box_sizes = tf.split(feature_map, [1, 2, 2], axis=-1)

    box_centers = box_centers + xy_offset
    box_centers = box_centers * ratio[::-1]

    box_sizes = tf.exp(box_sizes) * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    return xy_offset, boxes, conf_logits


def compute_loss(y_pred, y_true):
    # Compute loss each layer.
    N = tf.cast(tf.shape(y_pred)[0], tf.float32)
    grid_size = tf.shape(y_pred)[1:3]
    image_size = tf.cast([image_h, image_w], tf.int32)
    ratio = tf.cast(image_size / grid_size, tf.float32)

    xy_offset, pred_boxes, conf_logits = reorg(y_pred)

    object_mask = y_true[..., 4:5]  # [N, 26, 16, 1]
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_cond(idx, ignore_mask):
        return tf.less(idx, tf.cast(N, tf.int32))

    def loop_body(idx, ignore_mask):
        valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
        iou = box_iou(pred_boxes[idx], valid_true_boxes)
        best_iou = tf.reduce_max(iou, axis=-1)  # [26, 16, 1]

        ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx + 1, ignore_mask

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)  # [N, 26, 16, 1]

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
    pred_boxes_xy = tf.expand_dims(pred_boxes_xy, -2)  # [26, 16, 1, 2]
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

    # preprocessing.
    create_dataset()

    is_training = tf.placeholder_with_default(False, shape=None, name='is_training')
    df_train = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    df_train.dropna(inplace=True)
    df_train.reset_index(inplace=True, drop=True)

    train_batch_num = int(math.ceil(float(len(df_train)) / batch_size))
    train_dataset = tf.data.Dataset.from_tensor_slices(df_train.values)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, True],
                             Tout=[tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=16
    )
    train_dataset = train_dataset.shuffle(len(df_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)

    image, y_true_l, y_true_s = iterator.get_next()
    image.set_shape([None, None, None, 3])
    y_true_l.set_shape([None, None, None, None])
    y_true_s.set_shape([None, None, None, None])

    y_pred_l, y_pred_s = ocr_network(image, is_training=is_training)
    loss_xy_l, loss_wh_l, loss_conf_l = compute_loss(y_pred_l, y_true_l)
    loss_xy_s, loss_wh_s, loss_conf_s = compute_loss(y_pred_s, y_true_s)

    total_loss = loss_xy_l + loss_wh_l + loss_conf_l + loss_xy_s + loss_wh_s + loss_conf_s

    pred_boxes, pred_confs = predict([y_pred_l, y_pred_s])
    pred_boxes, pred_confs = gpu_nms(pred_boxes, pred_confs, num_classes=1, score_thresh=0.3, nms_thresh=0.45)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        grads_and_vars = opt.compute_gradients(total_loss)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

    # validation.
    val_batch_num = 4150
    val_dataset = tf.data.Dataset.from_tensor_slices(os.listdir(os.path.join(input_dir, 'test_images')))
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_val_data,
                             inp=[x, True],
                             Tout=[tf.float32, tf.string]),
        num_parallel_calls=16
    )
    val_dataset = val_dataset.shuffle(4150)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)
    val_init_op = val_iterator.make_initializer(val_dataset)
    val_image, val_image_id = val_iterator.get_next()
    val_image.set_shape([None, None, None, 3])
    val_image_id.set_shape((None,))

    val_pred_l, val_pred_s = ocr_network(val_image, is_training=is_training)
    val_boxes, val_confs = predict([val_pred_l, val_pred_s])
    val_boxes, val_confs = gpu_nms(val_boxes, val_confs, num_classes=1, score_thresh=0.3, nms_thresh=0.45)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Training section.
    print('Training start.')
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        for epoch in range(30):
            sess.run(train_init_op)
            for i in range(train_batch_num):
                _, _total_loss, _pred_boxes, _pred_confs, _global_step = sess.run(
                    [train_op, total_loss, pred_boxes, pred_confs, global_step],
                    feed_dict={is_training: True})

                if _global_step % 1000 == 0:
                    print('global step: {}\ttotal loss: {}\tbox_num: {}'.format(_global_step, _total_loss,
                                                                                len(_pred_boxes)))
                    print(_pred_boxes.shape)
                    saver.save(sess, 'ocr_model.ckpt')

        sess.run(val_init_op)
        for i in range(val_batch_num):
            _val_boxes, _val_confs, _val_image_id = sess.run([val_boxes, val_confs, val_image_id])
            result_img = visualize(
                input_dir + 'test_images/' + str(_val_image_id[0].decode()) + '.jpg',
                _val_boxes, _val_confs)
            plt.figure(figsize=(15, 15))
            plt.imshow(result_img)
            plt.show()
