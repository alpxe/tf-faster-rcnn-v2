# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import com.util.tool_util as tool

from com.net.vggnet import vggnet
from com.net.snippets import Anchor, Proposal
from com.net.snippets.Proposal import proposal_layer

from com.net.snippets.anchors_building import generate_anchors_func


class network:
    def __init__(self, dataset, is_training=True):
        self.is_training = is_training

        self._net_info = {}

        self._classes = ('__background__',  # always index 0
                         'JJY')
        self._num_classes = 2

        self.image_width = dataset["width"]
        self.image_height = dataset["height"]

        # 位置的百分比 需要乘上宽/高 才能具体出数字
        gt_x1 = tf.multiply(dataset["xmin"], tf.cast(self.image_width, tf.float32))
        gt_y1 = tf.multiply(dataset["ymin"], tf.cast(self.image_height, tf.float32))
        gt_x2 = tf.multiply(dataset["xmax"], tf.cast(self.image_width, tf.float32))
        gt_y2 = tf.multiply(dataset["ymax"], tf.cast(self.image_height, tf.float32))
        gt_label = tf.cast(dataset["label"], tf.float32)
        self.gt_boxes = tf.to_float(tf.stack([gt_x1, gt_y1, gt_x2, gt_y2, gt_label], axis=1))

        # 锚框相关
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

    def _anchor_target_layer(self, rpn_cls_score):
        with tf.variable_scope("ANCHOR_TAR"):
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                Anchor.anchor_target_func,
                [rpn_cls_score, self.gt_boxes, self.image_width, self.image_height, self.anchors, self.K],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self.K * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self.K * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self.K * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._net_info['rpn_labels'] = rpn_labels
            self._net_info['rpn_bbox_targets'] = rpn_bbox_targets
            self._net_info['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._net_info['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights
        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                Proposal.proposal_target_func,
                [rois, roi_scores, self.gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([None, 5])
            roi_scores.set_shape([None])
            labels.set_shape([None, 1])
            bbox_targets.set_shape([None, self._num_classes * 4])
            bbox_inside_weights.set_shape([None, self._num_classes * 4])
            bbox_outside_weights.set_shape([None, self._num_classes * 4])

            self._net_info['rois'] = rois
            self._net_info['labels'] = tf.to_int32(labels, name="to_int32")
            self._net_info['bbox_targets'] = bbox_targets
            self._net_info['bbox_inside_weights'] = bbox_inside_weights
            self._net_info['bbox_outside_weights'] = bbox_outside_weights

        return rois, roi_scores

    def _rpn(self, net, initializer):
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
            rois, roi_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred,
                                              self.image_width, self.image_height, self.anchors, self.K)

            # rpn_labels 值的意义 输出的是对应的锚框是正标签和负背景
            rpn_labels = self._anchor_target_layer(rpn_cls_score)

            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")

        self._net_info['rpn_cls_score'] = rpn_cls_score
        self._net_info['rpn_cls_score_reshape'] = rpn_cls_score_reshape
        self._net_info['rpn_bbox_pred'] = rpn_bbox_pred

        return rois

    def _crop_pool_layer(self, conv, rois, name):
        with tf.variable_scope(name):
            # [0,0,0.....] #begin[0,0]  #[-1,1] >> 第一个维度 -1:取所有 ， 第二个维度 1:取一个
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])

            conv_shape = tf.shape(conv)  # (?, ?, ?, 512)
            # 原始图片的宽高
            height = (tf.to_float(conv_shape[1]) - 1.) * np.float32(self.stride)
            width = (tf.to_float(conv_shape[2]) - 1.) * np.float32(self.stride)

            # 建议框的坐标
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))

            POOLING_SIZE = 7
            # 截图
            crops = tf.image.crop_and_resize(conv, bboxes, tf.to_int32(batch_ids),
                                             [POOLING_SIZE, POOLING_SIZE], name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        # 2000 x 4096 -- 4096 x 2 ==> 2000x2
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = tool.softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")

        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')

        self._net_info["cls_score"] = cls_score
        self._net_info["cls_pred"] = cls_pred
        self._net_info["cls_prob"] = cls_prob
        self._net_info["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def build_network(self, net_conv):
        """
        创建网络
        """
        # 正态分布
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)

        with tf.variable_scope("build_network"):
            rois = self._rpn(net_conv, initializer)

            pool5 = self._crop_pool_layer(net_conv, rois, "pool5")

        fc7 = vggnet.full(pool5, self.is_training)  # fc7 2000x[4096个参]
        self._region_classification(fc7, self.is_training, initializer, initializer_bbox)

    def losses(self):
        with tf.variable_scope('LOSS'):
            rpn_cls_score = tf.reshape(self._net_info['rpn_cls_score_reshape'], [-1, 2])  # 2K -> [ 前景 , 背景 ]
            rpn_labels = tf.reshape(self._net_info['rpn_labels'], [-1])  # RPN_Label
            rpn_select = tf.where(tf.not_equal(rpn_labels, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])  # 正负标记

            # RPN 分类LOSS
            rpn_cross_entropy = tf.reduce_mean(  # 前景：[?,?]与[0,1] 交叉熵   背景：[?,?]与[1.0]交叉熵
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

            # RPN, bbox loss
            rpn_bbox_pred = self._net_info['rpn_bbox_pred']  # 4K -> Δ(x,y,w,h)
            rpn_bbox_targets = self._net_info['rpn_bbox_targets']  # RPN-> Δ(x,y,w,h)
            rpn_bbox_inside_weights = self._net_info['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._net_info['rpn_bbox_outside_weights']

            # RPN 回归LOSS
            rpn_loss_box = tool.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                               rpn_bbox_outside_weights, sigma=3.0, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._net_info["cls_score"]
            label = tf.reshape(self._net_info["labels"], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # RCNN, bbox loss
            bbox_pred = self._net_info['bbox_pred']
            bbox_targets = self._net_info['bbox_targets']
            bbox_inside_weights = self._net_info['bbox_inside_weights']
            bbox_outside_weights = self._net_info['bbox_outside_weights']
            loss_box = tool.smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses().append(0), 'regu')

        return loss + regularization_loss
