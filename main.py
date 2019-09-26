# -*- coding: utf-8 -*-

import tensorflow as tf
import platform
import cv2
import numpy as np

from com.dataset.tfrecords import Records
from com.net.vggnet import vggnet
from com.net.network import network

print("Python version: {0}".format(platform.python_version()))
print("TensroFlow version: {0}\n".format(tf.__version__))

is_training = True

# 该目录下，存放 图片与真实标记的绑定框与标签 image与xml对应
# 我所使用的是 LabelImg 这款小软件 (安装后,终端$LabelImg 回车即可打开)
base_image_path = "resource/train"

# csv文件生成的路径
csv_path = "resource/csv/JJY_train.csv"

# tfrecord文件的路径
tf_path = "resource/TFR/train.tfrecord"

# vgg_16的ckpt模型，已经训练好对图片敏感的参数
# https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
vgg_model_path = "ckpt/vgg16/vgg_16.ckpt"

tfr = Records()

# 生成xml
tfr.xml2csv(base_image_path, csv_path)

# 生成tfrecords
tfr.generate(csv_path, tf_path, base_image_path)

data = tfr.extract(tf_path)  # 从数据集中提取数据

img = data["image"]
height = data["height"]
width = data["width"]

vgg = vggnet(is_training=is_training)
map = vgg.vgg16(img)  # 经过vgg16 -> 特征图

net = network(data, is_training=is_training)
net.build_network(map)
loss = net.losses()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(loss))
    # for i in range(10):

    # res_img, resa = sess.run([img, anchors])
    # res_img = res_img[0]
    #
    # print(len(resa))
    # for item in resa:
    #     # print(item)
    #     x1 = np.int(item[1])
    #     y1 = np.int(item[2])
    #     x2 = np.int(item[3])
    #     y2 = np.int(item[4])
    #
    #     cv2.rectangle(res_img, (x1, y1), (x2, y2), (126, 0, 0), 1)
    #
    # cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    # cv2.imshow("camera", res_img)
    # cv2.waitKey()
