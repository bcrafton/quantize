

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--nexamples', type=int, default=1024)
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

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def quantize_np(x):
  scale = 127 / np.max(np.absolute(x))
  x = x * scale
  x = np.round(x)
  x = np.clip(x, -127, 127)
  return x, scale

###############################################################

weights = np.load('imagenet224.npy', allow_pickle=True).item()

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

# pytorch resnet 18:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# pytorch quantization: 
# https://pytorch.org/blog/introduction-to-quantization-on-pytorch/

# x_process = tf.keras.applications.resnet50.preprocess_input(x_process)
# x_process = x_process / tf.math.reduce_std(x_process)

def evaluate(x, y, scale):
    model_predict = m.train(x, scale)
    predict = tf.argmax(model_predict, axis=1)
    actual = tf.argmax(y, 1)
    correct = tf.equal(predict, actual)
    sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
    return sum_correct

def collect(x, n):
    model_predict = m.upto(x, n)
    return model_predict

def get_weights():
    return m.get_weights()

##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']

total_correct = 0
for jj in range(0, args.nexamples, args.batch_size):
    s = jj
    e = jj + args.batch_size
    correct = evaluate(xs[s:e].astype(np.float32), ys[s:e], False)
    total_correct += correct

acc = total_correct / args.nexamples
print (acc)

##################################################################

m.set_ymax()
print (m.ymax)

##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']
xs, scale = quantize_np(xs)

for n in range(31):
    for jj in range(0, args.nexamples, args.batch_size):
        s = jj
        e = jj + args.batch_size
        p = collect(xs[s:e].astype(np.float32), n)
    
##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']
xs, scale = quantize_np(xs)

total_correct = 0
for jj in range(0, args.nexamples, args.batch_size):
    s = jj
    e = jj + args.batch_size
    correct = evaluate(xs[s:e].astype(np.float32), ys[s:e], True)
    total_correct += correct

acc = total_correct / args.nexamples
print (acc)

##################################################################

# weight_dict = sess.run(weights, feed_dict={})

weight_dict = get_weights()

for key in weight_dict.keys():
    print (weight_dict[key].keys())

# weight_dict['acc'] = acc
np.save('resnet18_quant_weights', weight_dict)

##################################################################








