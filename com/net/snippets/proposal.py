# -*- coding: utf-8 -*-
import tensorflow as tf
from com.net.snippets.bbox_transform import bbox_transform_inv_tf, clip_boxes


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, width, height, anchors, K):
    post_nms_topN = 2000
    nms_thresh = 0.7

    # 获取分数和边界框

    # K=9 1个K的意思就是：一个锚点有9个框
    # CNN出 2K 和 4K  2K:就有18个量，即1个框有2个量(类别)  同理1个框有4个量(坐标)
    # 这里的分数，即[:,:,:,9:] -> 取出的量是 18个量中 后9个量。 也就是说这后9个量 表示这9个锚框的前景打分值
    scores = rpn_cls_prob[:, :, :, K:]  # 前景打分概率值
    scores = tf.reshape(scores, shape=(-1,))
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    # 把anchors框通过 增量Delta 转换为proposals
    # 也就是说，输入anchors和计算出来的dw dh dx dy，计算得到修正后的proposal
    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, width, height)  # 去除超出边界的框

    # Non-maximal suppression 非极大值抑制
    # 预测框,对应框的预测分数, topN取前2000 IoU阈值0.7
    indices = tf.image.non_max_suppression(proposals, scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

    # 索引
    boxes = tf.gather(proposals, indices)  # 来自索引
    boxes = tf.to_float(boxes)
    scores = tf.gather(scores, indices)
    scores = tf.reshape(scores, shape=(-1, 1))

    #  Only support single image as input
    # tf.shape(indices)[0] -> 应该是post_nms_topN的值 2000
    batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)  # ->shape:(2000,1)
    blob = tf.concat([batch_inds, boxes], 1)  # 一组[ 0 ,x1,y1,x2,y2]

    blob.set_shape([None, 5])
    scores.set_shape([None, 1])

    # [ 0 ,x1,y1,x2,y2] ~ [评分]
    return blob, scores
