# -*- coding: utf-8 -*-
import tensorflow as tf

import numpy as np
import numpy.random as npr

from com.so.bbox.cython_bbox import bbox_overlaps

from com.net.snippets.bbox_transform import bbox_transform_inv_tf, clip_boxes, bbox_transform


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, width, height, anchors, K):
    """
    （bbox_transform_inv_tf） anchor + 增量偏移估计值 => 预测框
    （clip_boxes） 坐标进行裁剪
    （tf.image.non_max_suppression）  非极大值抑制 （训练时2000个，测试时300个）

    :param rpn_cls_prob: 降维为2K 的特征数值 表示二分类-前景/背景
    :param rpn_bbox_pred: 降维为4K 的特征数值 表示坐标的4个值
    :param width: 图片本身的宽度 处理越界框
    :param height: 图片本身的高度
    :param anchors: 锚框 左上,右下
    :param K: K=9
    :return: 候选区域 第一列为全为0的batch_inds 后4列为坐标（坐上+右下）
    """
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


def proposal_target_func(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    """
    将目标检测的建议分配给真实框，
    提出建议 分类标签和边界框回归目标
    :param rpn_rois:  [0,x1,y1,x2,y2] 经过非极大化抑制所筛选出来的优质框 大约2000个
    :param rpn_scores:  [[?],[?]...[0]] 与框对应的 前景分数值
    :param gt_boxes:  [x1,y1,x2,y2,label]
    :param _num_classes:
    :return:
    """

    all_rois = rpn_rois
    all_scores = rpn_scores

    # """
    # 将 gt_boxes[x1,y1,x2,y2,label] 转成 [ 0 ,x1,y1,x2,y2]
    # 拼入 all_rois -> [[0,x1,y1,x2,y2]...]
    # """
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
    all_scores = np.vstack((all_scores, zeros))

    num_images = 1
    BATCH_SIZE = 128  # Minibatch size (number of regions of interest [ROIs]) ROI感兴趣的数量
    FG_FRACTION = 0.25  # Fraction of minibatch that is labeled foreground (i.e. class > 0) 标记为前景的小批量的分数（即类> 0）

    rois_per_image = BATCH_SIZE / num_images  # 128 每个图片 感兴趣的区域数量  这里图片数是1
    fg_rois_per_image = np.round(FG_FRACTION * rois_per_image)  # 32 前景框占比得出个数

    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)

    rois = rois.reshape(-1, 5)  # 前后背景框 [0,x1,y1,x2,y2]
    roi_scores = roi_scores.reshape(-1)  # 前后背景框 对应的scores
    labels = labels.reshape(-1, 1)  # IoU最优的框 阈值大于设定值(0.5) 绑定框的labels 后面是背景的都强制设置为0了
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)  # 坐标 存入对应的类别位置中
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)  # 前景权重
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)  # 背景权重，目前是一致的

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """"""

    """
    计算提取框框与真实框的重合率 IoU
         Q
    [ P [?],
      P [?],
      P [?] ]   overlaps[P, Q]

    值得提一下，输入的rois是预测框(由预测打分值socre非极大值抑制而来)，因此预测框与绑定框会差的很远
    随着打分值的修正，非极大值抑制而得出的rois会越来越与绑定框靠近
    因此overlaps的值会越来越大(最大是1)
    all_rois中拼接了gt_box， 所以gt_box与gt_box IoU=1(自身与自身当然完美重合)
    """
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))

    gt_assignment = overlaps.argmax(axis=1)  # 取出其维度中最大值的索引 [0,0,0,...]
    max_overlaps = overlaps.max(axis=1)  # 取出其维度中最大值 [?,?,?,?,1] 注意：all_rois拼入了一个gt_box
    labels = gt_boxes[gt_assignment, 4]

    FG_THRESH = 0.5  # ROI的重叠阈值被视为前景
    BG_THRESH_HI = 0.5  # 小于0.5 大于0.1 视为背景
    BG_THRESH_LO = 0.1

    # 从 IoU值
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]  # 前景值索引
    bg_inds = np.where((max_overlaps < BG_THRESH_HI) &
                       (max_overlaps >= BG_THRESH_LO))[0]

    # 确保对固定数量的区域进行采样
    if fg_inds.size > 0 and bg_inds.size > 0:  # 前景和背景都存在
        # 预计前景框数量：128*0.25=32个
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        # 如果前景框小与预计数量，则前景框几个就取几个 如果大于预计数量 则随机挑出预计数量个数的前景框
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image  # 剩下的则是背景框的数量
        to_replace = bg_inds.size < bg_rois_per_image  # 如果背景框的数量比较少 则就重复提取 保持总数是128个
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    elif fg_inds.size > 0:  # 只存在前景
        to_replace = fg_inds.size < rois_per_image  # 如果前景框的数量不超过 128个 靠重复随机去填满128个
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image  # 前景样本数等于总样本数
    elif bg_inds.size > 0:  # 只存在背景框
        to_replace = bg_inds.size < rois_per_image  # 如果背景框的数量不超过 128个 靠重复随机去填满128个
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0  # 前景框样本数等于0
    else:
        print("不存在前后背景框，此错误应设置断点检查")

    # 前景索引+背景索引
    keep_inds = np.append(fg_inds, bg_inds)

    labels = labels[keep_inds]
    labels[int(fg_rois_per_image):] = 0  # 因为后面拼接的是背景 所以可以明确的将值设置为0

    rois = all_rois[keep_inds]  # 最优的前后背景 提取框
    roi_scores = all_scores[keep_inds]

    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data, num_classes)

    """
    Returns:
     labels:  [1,2,3,...0,0,0,0] IoU最优的框 阈值大于设定值(0.5) 绑定框的labels
     rois:  前后背景 提取框
     rois_scores:  对应的分数
     bbox_targets:   前景框数个 x [0,0,0,0  ,0,0,0,0  ,0,0,0,0  ,...] 4个一组 num_class几个就几组 对应的类别存在对应类别的位置
                     值为 Delta Δ
     bbox_inside_weights:  权重与bbox_targets 类似. 但是值为 ..., 1,1,1,1,  ...代表权重为1
    """
    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """
    计算图像边界回归框的目标
    :param ex_rois: 预测框
    :param gt_rois: 真实框
    :param labels: 真实标签
    :return:
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]  # 个数一致
    assert ex_rois.shape[1] == 4  # 4个坐标数值 x1,y1,x2,y2
    assert gt_rois.shape[1] == 4  # 4个坐标数值

    # 通过公式 当前的预测框 与其最密切的绑定框 之间的增量 计算出之间的差值 Delta Δ
    targets = bbox_transform(ex_rois, gt_rois)  # Delta Δ

    BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
    BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
    targets = ((targets - np.array(BBOX_NORMALIZE_MEANS)) / np.array(BBOX_NORMALIZE_STDS))

    # [label, dx, dy, dw, dh] 这里的label是 绑定框的类别值 0代表背景，1，2，3，4....表示对应某种东西的类别
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]  # 取出类别
    # 前景框个数 x  numclass个0000  即每组框坐标(0000) 对应1个类别 由下面for ind in inds 体现规划
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)

    inds = np.where(clss > 0)[0]  # 代表前景的索引

    # 对应的类别分配到对应位置的坐标组
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4

        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)  # 因为是前景值，所以权重是1
        pass

    return bbox_targets, bbox_inside_weights
