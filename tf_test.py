

import argparse
import os
import sys

##############################################

import keras
import tensorflow as tf
import numpy as np
import time
np.set_printoptions(threshold=1000)

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from layers import *

###############################################################

x = tf.placeholder(tf.float32, [1, 224, 224, 3])

weights = np.load('resnet18.npy', allow_pickle=True).item()

m = model(layers=[
conv_block((7,7,3,64), 2, noise=None, weights=weights),

max_pool(2, 3),

res_block1(64,   64, 1, noise=None, weights=weights),
res_block1(64,   64, 1, noise=None, weights=weights),

res_block2(64,   128, 2, noise=None, weights=weights),
res_block1(128,  128, 1, noise=None, weights=weights),

res_block2(128,  256, 2, noise=None, weights=weights),
res_block1(256,  256, 1, noise=None, weights=weights),

res_block2(256,  512, 2, noise=None, weights=weights),
res_block1(512,  512, 1, noise=None, weights=weights),

avg_pool(7, 7),
dense_block(512, 1000, noise=None, weights=weights)
])

y = m.train(x=x)
ys = m.collect(x=x)

weights = m.get_weights()

##################################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

##################################################################

xs = np.load('input_image.npy', allow_pickle=True)
xs = np.reshape(xs, (1,224,224,3))
[np_y, np_ys] = sess.run([y, ys], feed_dict={x: xs})
np_y = np_y[0]
# print (xs[0,0,0,:])
print (np.argmax(np_y))
print (np_y[np.argmax(np_y)])
print (np_y)

##################################################################

ref = np.load('ref.npy', allow_pickle=True)
print (np.std(ref - np_ys[0]))

import matplotlib.pyplot as plt
a = np_ys[0]
a = a[0, :, :, 1]
print (np.shape(a))
print (a[0])
plt.imshow(a)
plt.show()

##################################################################

[weights] = sess.run([weights], feed_dict={})

np.save('resnet18_weights', weights)










