# -*- coding: utf-8 -*-
import numpy as np
import numpy.random as npr

from com.so.bbox.cython_bbox import bbox_overlaps

from com.net.snippets.bbox_transform import bbox_transform
import com.util.tool_util as tool


def anchor_target_func(rpn_cls_score, gt_boxes, width, height, all_anchors, K):
    total_anchors = all_anchors.shape[0]  # 锚框的个数

    # 边框值
    _allowed_border = 0

    # map of shape (..., H, W)
    fh, fw = rpn_cls_score.shape[1:3]

    # 不越界的锚框索引
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < width + _allowed_border) &  # width
        (all_anchors[:, 3] < height + _allowed_border)  # height
    )[0]  # 第一维

    anchors = all_anchors[inds_inside, :]  # 不越界的锚框

    # 初始化 labels
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # -1表示忽略

    # labels的长度 与 anchor(P)的个数相关
    # 计算锚框与真实框的重合率 IoU
    # 二维表格： ，竖着是anchors 横向是gt_bboxes
    """
    [ P [Q Q Q Q],
      P [Q Q Q Q],
      P [Q Q Q Q] ]   overlaps[P, Q]
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, np.float),
        np.ascontiguousarray(gt_boxes, np.float)
    )

    # ps: argmax_overlaps 这里我的bbox就一个，所以横最大的始终是索引0 即值为0
    argmax_overlaps = overlaps.argmax(axis=1)  # 横向压缩， 每个的anchors最优的gt_boxes 的索引
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # 通过索引，找到每个anchors对应的具体值(IOU值)

    gt_argmax_overlaps = overlaps.argmax(axis=0)  # 竖向压缩， 每个gt_boxes最优的anchor 的索引
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]  # 通过索引，找到每个gt_boxes对应的具体值

    # np.where 根据维度的返回数组，数组表示  第0维的第几个， 第1维度的第几个， 第N维的第几个
    # 这里取[0] 表示只关心 竖列的值 不关心它是在这个竖列中的第几个。  竖列表示每个anchors
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  # 即可得到 gt_boxes 唯一对应的那个anchor 索引

    # 如果max_overlaps的值小于0.3 将对应的lable值设置0 ：背景
    labels[max_overlaps < 0.3] = 0

    # gt_boxes对应最优的anchors位置 设置成前景。 矮个中找最高也要设置成1
    labels[gt_argmax_overlaps] = 1

    # IoU值大于0.7 设置成1：前景
    labels[max_overlaps >= 0.7] = 1

    # 平衡正负样本数 -------

    # 正样预期本个数
    num_fg = int(256 * 0.5)  # 类型一致

    # 当前正样本的索引
    fg_inds = np.where(labels == 1)[0]

    if len(fg_inds) > num_fg:  # 正样本数 大于 预设的正样本数
        # 随机抽出 多出的样本数量 的索引  将超出预期数的正样本随机忽略
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
        pass

    # 负样本的个数  经过上一步if判断后，labels正样本的个数 只会小于128
    num_bg = 256 - np.sum(labels == 1)

    # 从labels中获取 负样本的索引
    bg_inds = np.where(labels == 0)[0]

    if len(bg_inds) > num_bg:
        # 将超出预期数的负样本随机忽略
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 平衡正负样本数 ------ labels 完成

    # 制作绑定框 bbox_targets
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 初始化
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 初始化
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)  # 初始化

    # gt_boxes[argmax_overlaps, :] : 最匹配的 get_boxes 所以这两个参数 长度一样
    # 通过公式 当前的anchor与其最密切的绑定框之间的差距 计算出之间的差值 Delta Δ
    bbox_targets = bbox_transform(anchors, gt_boxes[argmax_overlaps, :]).astype(np.float32, copy=False)

    # 样本权重
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))

    labels = tool.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = tool.unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = tool.unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = tool.unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((1, fh, fw, K)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, K * fh, fw))  # 不懂为什么要reshape成这个结构
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, fh, fw, K * 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, fh, fw, K * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, fh, fw, K * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
