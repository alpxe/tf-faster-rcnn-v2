# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def forecast_box(reg_bbox, anchors):
    """
    :param reg_bbox:
    :param anchors:
    :return: 预测框
    """
    reg_bbox = tf.reshape(reg_bbox, shape=(-1, 4))

    abox = tf.cast(anchors, reg_bbox.dtype)  # 让这连个数据类型一致

    # 将左上右下坐标 转成 x,y,w,h
    aw = tf.subtract(abox[:, 2], abox[:, 0]) + 1.0
    ah = tf.subtract(abox[:, 3], abox[:, 1]) + 1.0
    ax = tf.add(abox[:, 0], aw * 0.5)
    ay = tf.add(abox[:, 1], ah * 0.5)

    # 将4K里的值 做Delta Δ，表示增量
    dx = reg_bbox[:, 0]
    dy = reg_bbox[:, 1]
    dw = reg_bbox[:, 2]
    dh = reg_bbox[:, 3]

    # 已知anchor值，delta值 将论文中的公式 变形后 输出预测框 x,y,w,h
    x = tf.add(tf.multiply(dx, aw), ax)
    y = tf.add(tf.multiply(dy, ah), ay)
    w = tf.multiply(tf.exp(dw), aw)
    h = tf.multiply(tf.exp(dh), ah)

    # 将x,y,w,h转化回 左上右下坐标
    pred_boxes0 = tf.subtract(x, w * 0.5)
    pred_boxes1 = tf.subtract(y, h * 0.5)
    pred_boxes2 = tf.add(x, w * 0.5)
    pred_boxes3 = tf.add(y, h * 0.5)

    # 输出 box的左上右下坐标
    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    """
    矩形 + Delta = 新的矩形
    :param boxes: 矩形框 左上,右下(x1,y1,x2,y2)
    :param deltas: Delta Δ，表示增量
    :return:
    """
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]  # 从0开始搜索，步长4
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_transform_inv_tf(boxes, deltas):
    boxes = tf.cast(boxes, deltas.dtype)
    widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
    heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
    ctr_x = tf.add(boxes[:, 0], widths * 0.5)
    ctr_y = tf.add(boxes[:, 1], heights * 0.5)

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
    pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
    pred_w = tf.multiply(tf.exp(dw), widths)
    pred_h = tf.multiply(tf.exp(dh), heights)

    pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
    pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
    pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
    pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

    return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)


def clip_boxes(boxes, w, h):
    """
    将超出范围的框 超出的部分裁剪
    :param boxes:
    :param w: 原始图像的宽
    :param h: 原始图像的高
    :return:
    """
    w = tf.cast(w, dtype=tf.float32)
    h = tf.cast(h, dtype=tf.float32)

    b0 = tf.maximum(tf.minimum(boxes[:, 0], w - 1), 0)
    b1 = tf.maximum(tf.minimum(boxes[:, 1], h - 1), 0)
    b2 = tf.maximum(tf.minimum(boxes[:, 2], w - 1), 0)
    b3 = tf.maximum(tf.minimum(boxes[:, 3], h - 1), 0)
    return tf.stack([b0, b1, b2, b3], axis=1)
