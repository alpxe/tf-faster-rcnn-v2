# -*- coding: utf-8 -*-

import tensorflow as tf

a = tf.constant([0.])

lost = tf.losses.get_regularization_losses()


print(lost.append(0))
print(lost)
with tf.Session() as sess:
    pass
