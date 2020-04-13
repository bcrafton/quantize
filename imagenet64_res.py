

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

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

###############################################################

weights = np.load('resnet18_quant.npy', allow_pickle=True).item()

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

def evaluate(x, y):
    model_predict = tf.nn.softmax(m.train(x))
    predict = tf.argmax(model_predict, axis=1)
    actual = tf.argmax(y, 1)
    correct = tf.equal(predict, actual)
    sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
    return predict, actual, sum_correct

##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']

xs = xs / 255. 
xs = xs - np.array([0.485, 0.456, 0.406])
xs = xs / np.array([0.229, 0.224, 0.225])

total_correct = 0
for jj in range(0, 1024, args.batch_size):
    s = jj
    e = jj + args.batch_size
    y, yhat, correct = evaluate(xs[s:e].astype(np.float32), ys[s:e])
    total_correct += correct
    # print (y, yhat)

acc = total_correct / 1024
print (acc)

##################################################################



