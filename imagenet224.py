
import argparse
import os
import sys
import time

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='imagenet224')
args = parser.parse_args()

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from bc_utils.conv_utils import conv_output_length
from bc_utils.conv_utils import conv_input_length

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from layers import *

import matplotlib.pyplot as plt

from load import Loader

##############################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

##############################################

# this might fail because we dont fuse the conv + bn first
# weights = np.load('resnet18.npy', allow_pickle=True).item()

# this might work because we fuse the conv + bn first
weights = np.load('resnet18_quant.npy', allow_pickle=True).item()

##############################################

# I think this works because we have (mean=0, std=1)
std = np.array([0.229, 0.224, 0.225]) * 255. / 2.
weights[0]['f'] = weights[0]['f'] / np.reshape(std, (3,1))

mean = np.array([0.485, 0.456, 0.406]) * 255. / 2.
expand_mean = np.ones(shape=(7,7,3)) * mean
expand_mean = expand_mean.flatten()

weights[0]['b'] = weights[0]['b'] - (expand_mean @ np.reshape(weights[0]['f'], (7*7*3, 64)))

##############################################

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

###############################################################

load = Loader('/home/brian/Desktop/ILSVRC2012/val')

total = 50000
batch_size = 50

accum_correct = 0
accum = 0

start = time.time()
for batch in range(0, total, batch_size):
    while load.empty(): pass
    
    x, y = load.pop()
    model_predict = m.predict(x)
    pred = np.argmax(model_predict.numpy(), axis=1)
    
    correct = np.sum(y == pred)
    accum_correct += correct
    accum += batch_size

    if (accum % 5000) == 0:
        print (accum / (time.time() - start), accum, accum_correct / accum)

load.join()

##################################################################

load = Loader('/home/brian/Desktop/ILSVRC2012/val')

total = 50000
batch_size = 50

accum_correct = 0
accum = 0

target = 1

start = time.time()
for batch in range(0, total, batch_size):
    while load.empty(): pass
    
    x, y = load.pop()
    model_predict = m.predict(x, q=True, l=target)
    pred = np.argmax(model_predict.numpy(), axis=1)
    
    correct = np.sum(y == pred)
    accum_correct += correct
    accum += batch_size

    if (accum % 1250) == 0:
        print (accum / (time.time() - start), target, accum, accum_correct / accum)
        target += 1

load.join()

##################################################################

load = Loader('/home/brian/Desktop/ILSVRC2012/val')

total = 50000
batch_size = 50

accum_correct = 0
accum = 0

start = time.time()
for batch in range(0, total, batch_size):
    while load.empty(): pass
    
    x, y = load.pop()
    model_predict = m.predict(x, q=True, l=100)
    pred = np.argmax(model_predict.numpy(), axis=1)
    
    correct = np.sum(y == pred)
    accum_correct += correct
    accum += batch_size

    if (accum % 5000) == 0:
        print (accum / (time.time() - start), accum, accum_correct / accum)

load.join()

##################################################################




