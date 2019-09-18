# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

import com.util.tool_util as tool

from com.net.snippets.anchors_building import generate_anchors_func
from com.net.snippets.proposal import proposal_layer


class network:
    def __init__(self, dataset, is_training=True):
        self.is_training = is_training

        self._net_info = {}

        self.image_width = dataset["width"]
        self.image_height = dataset["height"]

        self.stride = 16  # 下采样
        self.anchor_scales = 2 ** np.arange(3, 6)
        self.anchor_ratios = [0.5, 1, 2]
        self.K = len(self.anchor_scales) * len(self.anchor_ratios)  # 3x3=9

        self.feature_width = tf.to_int32(tf.ceil(self.image_width / self.stride))  # 特征图宽度 来自tfrecords
        self.feature_height = tf.to_int32(tf.ceil(self.image_height / self.stride))  # 特征图高度
        self.anchors = self.generate_anchors()

    def generate_anchors(self):
        """
        生成锚框
        :return:
        """

        with tf.variable_scope("ANCHOR_GEN"):
            _anchors, _anchor_length = tf.py_func(generate_anchors_func,
                                                  [self.feature_width, self.feature_height,
                                                   self.stride, self.anchor_scales, self.anchor_ratios],
                                                  [tf.float32, tf.int32], name="generate_anchors")

            _anchors.set_shape([None, 4])

            return _anchors

    def rpn(self, net, initializer):
        # 将网络进行3x3卷积，得到一个共享网络层
        rpn = slim.conv2d(net, 256, [3, 3], trainable=self.is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")

        # rpn分类的得分值  rpn_cls_score 与 rpn_label 进行loss
        rpn_cls_score = slim.conv2d(rpn, 2 * self.K, [1, 1], trainable=self.is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')

        # rpn绑定框预测值 detail值 rpn_bbox_pred 与 rpn_bbox_targets 进行loss
        rpn_bbox_pred = slim.conv2d(rpn, 4 * self.K, [1, 1], trainable=self.is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        rpn_cls_score_reshape = tool.reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = tool.softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        rpn_cls_prob = tool.reshape_layer(rpn_cls_prob_reshape, 2 * self.K, "rpn_cls_prob")  # 归一化概率

        if self.is_training:
            rois, rpn_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred,
                                              self.image_width, self.image_height, self.anchors, self.K)

        return rois

    def build_network(self, net_conv):
        """
        创建网络
        """
        # 正态分布
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

        with tf.variable_scope("build_network"):
            return self.rpn(net_conv, initializer)
            pass
