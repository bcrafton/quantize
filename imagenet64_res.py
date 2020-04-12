

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

##############################################

import keras
import tensorflow as tf
import numpy as np
import time
np.set_printoptions(threshold=1000)

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from layers import *

MEAN = [122.77093945, 116.74601272, 104.09373519]

###############################################################

x = tf.placeholder(tf.float32, [args.batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32, [args.batch_size, 1000])

weights = np.load('resnet18.npy', allow_pickle=True).item()

# 2 things:
# > relu 6
# > use mean/var from model
# > why are they missing relus ???

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

# very helpful:
# https://pytorch.org/hub/pytorch_vision_resnet/

x_process = x
# x_process = tf.keras.applications.resnet50.preprocess_input(x_process)
# x_process = x_process / tf.math.reduce_std(x_process)
model_predict = tf.nn.softmax(m.train(x=x_process))

predict = tf.argmax(model_predict, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##################################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']

# xs = xs - np.mean(xs, axis=(0,1,2))
# xs = xs / np.std(xs, axis=(0,1,2))
xs = xs / np.max(xs, axis=(0,1,2))
xs = xs - np.array([0.485, 0.456, 0.406])
xs = xs / np.array([0.229, 0.224, 0.225])

total_correct = 0
for jj in range(0, 1024, args.batch_size):
    s = jj
    e = jj + args.batch_size
    [np_sum_correct] = sess.run([sum_correct], feed_dict={x: xs[s:e], y: ys[s:e]})
    total_correct += np_sum_correct

acc = total_correct / 1024
print (acc)

##################################################################



