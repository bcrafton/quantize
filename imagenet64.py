

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

def quantize_np(x, low, high):
  scale = (np.max(x) - np.min(x)) / (high - low)
  x = x / scale
  x = np.floor(x)
  x = np.clip(x, low, high)
  return x

###############################################################

weights = np.load('resnet18_quant.npy', allow_pickle=True).item()

# 2 things:
# > relu 6
# > use mean/var from model
# > why are they missing relus ???

m = model(layers=[
conv_block((7,7,3,64), 2, noise=None, weights=weights),
])

# very helpful:
# https://pytorch.org/hub/pytorch_vision_resnet/

# pytorch resnet 18:
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# pytorch quantization: 
# https://pytorch.org/blog/introduction-to-quantization-on-pytorch/

# x_process = tf.keras.applications.resnet50.preprocess_input(x_process)
# x_process = x_process / tf.math.reduce_std(x_process)

def evaluate(x):
    model_predict = m.train(x)
    return model_predict

def get_weights():
    return m.get_weights()

##################################################################

dataset = np.load('val_dataset.npy', allow_pickle=True).item()
xs, ys = dataset['x'], dataset['y']

xs = xs / 255. 
xs = xs - np.array([0.485, 0.456, 0.406])
xs = xs / np.array([0.229, 0.224, 0.225])
xs = quantize_np(xs, -127, 127)

##################################################################

xs = np.reshape(xs[0], (1,224,224,3)).astype(np.float32)
y = evaluate(xs)

print (np.max(y.numpy()))

##################################################################



