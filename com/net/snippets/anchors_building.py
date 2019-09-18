# -*- coding: utf-8 -*-
import numpy as np


def generate_anchors_func(fw, fh, stride, anchor_scales, anchor_ratios):
    """
    创建锚框
    :param fw:
    :param fh:
    :param stride:
    :param anchor_scales:
    :param anchor_ratios:
    :return:
    """
    # meshgrid 组合尺寸和比例
    scales, ratios = np.meshgrid(anchor_scales, anchor_ratios)
    scales, ratios = scales.flatten(), ratios.flatten()  # 平铺

    # 锚框的 宽与高
    scalesX = scales * np.sqrt(ratios)
    scalesY = scales / np.sqrt(ratios)

    shiftsX = np.arange(0, fw) * stride
    shiftsY = np.arange(0, fh) * stride

    # meshgrid 组合中心坐标  想象成二维表格
    shiftsX, shiftsY = np.meshgrid(shiftsX, shiftsY)

    # 锚框的x坐标对应着锚框的宽
    centerX, anchorX = np.meshgrid(shiftsX, scalesX)
    # 锚框的y坐标对应着锚框的高
    centerY, anchorY = np.meshgrid(shiftsY, scalesY)

    # stack 各种尺寸，各种比例对应各种长度
    anchor_center = np.stack([centerX, centerY], axis=2).reshape(-1, 2)
    anchor_size = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)

    # 左上 右下 坐标输出
    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)
    # 类型转换
    boxes = boxes.astype(np.float32)
    length = np.int32(boxes.shape[0])

    return boxes, length
