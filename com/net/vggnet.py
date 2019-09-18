# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow


class vggnet:
    def __init__(self, is_training=True):
        self.is_training = is_training
        pass

    def vgg16(self, inputX):
        with tf.variable_scope("vgg_16"):
            net = slim.repeat(inputX, 2, slim.conv2d, 64, [3, 3], trainable=False, scope="conv1")
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=self.is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=self.is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')

            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=self.is_training, scope='conv5')

            return net

    def __get_variables_to_resotre(self, vgg_path):
        """
        Get the variables to restore, ignoring the variables to fix
        获取要恢复的变量，忽略要修复的变量

        :param variables: 由 tf.global_variables() 获得 自己网络图中的所有变量
        :param var_keep_dic:
        :return:
        """
        # 获取 模型中的变量
        reader = pywrap_tensorflow.NewCheckpointReader(vgg_path)
        var_keep_dic = reader.get_variable_to_shape_map()

        variables_to_restore = []
        _variables_to_fix = {}

        variables = tf.global_variables()
        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == ('vgg_16/fc6/weights:0') or \
                    v.name == ('vgg_16/fc7/weights:0'):
                _variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == ('vgg_16/conv1/conv1_1/weights:0'):
                _variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                # print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore, _variables_to_fix

    def restore(self, sess, vgg_path):
        """
        恢复vgg ckpt 模型数据

        :param sess:
        :param vgg_path:
        :return:
        """
        # res 需要恢复， fix 需要忽略
        res, fix = self.__get_variables_to_resotre(vgg_path)

        restorer = tf.train.Saver(res)
        restorer.restore(sess, vgg_path)
        print('vgg res >> Loaded.')

        # 需要在加载前修复变量，以便将RGB权重更改为BGR
        # 对于VGG16，它还将卷积权重fc6和fc7更改为完全连接的权重
        # fix_variables(sess, vgg_path, fix)
        # print('vgg res >> Fixed.')
        pass

    pass
