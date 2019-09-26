# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def reshape_layer(bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name):
        # change the channel to the caffe format
        to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
        # then force it to have channel 2
        reshaped = tf.reshape(to_caffe,
                              tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
        # then swap the channel back
        to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf


def softmax_layer(bottom, name):
    with tf.variable_scope(name):
        input_shape = tf.shape(bottom)
        bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        return tf.reshape(reshaped_score, input_shape)


def smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    """

    :param bbox_pred:
    :param bbox_targets:
    :param bbox_inside_weights:
    :param bbox_outside_weights:
    :param sigma: 西格玛
    :param dim:
    :return:
    """
    sigma_2 = sigma ** 2  # σ^2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff  # 公式中的 X
    abs_in_box_diff = tf.abs(in_box_diff)  # 绝对值

    # tf.less返回两个张量各元素比较（x<y）得到的真假值组成的张量
    # 公式中的判断 if |x|< 1/σ^2
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))

    # 公式
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
    return loss_box


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:  # 如果是一维数组
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret
