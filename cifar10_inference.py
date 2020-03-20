
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--noise', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--name', type=str, default="cifar10_weights")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import *
from layers import *

####################################

def quantize_np(x, low, high):
  scale = (np.max(x) - np.min(x)) / (high - low)
  x = x / scale
  x = np.floor(x)
  x = np.clip(x, low, high)
  return x
  
####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
# x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train = quantize_np(x_train, 0, 127)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
# x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test = quantize_np(x_test, 0, 127)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()

m = model(layers=[
conv_block(3,   64, 1, noise=args.noise, weights=weights[0]),
conv_block(64,  64, 2, noise=args.noise, weights=weights[1]),

conv_block(64,  128, 1, noise=args.noise, weights=weights[2]),
conv_block(128, 128, 2, noise=args.noise, weights=weights[3]),

conv_block(128, 256, 1, noise=args.noise, weights=weights[4]),
conv_block(256, 256, 2, noise=args.noise, weights=weights[5]),

avg_pool(4, 4, weights=weights[6]),
dense_block(256, 10, noise=args.noise, weights=weights[7])
])

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

model_predict = m.inference(x=x)

####################################

predict = tf.argmax(model_predict, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

total_correct = 0
for jj in range(0, 10000, args.batch_size):
    s = jj
    e = jj + args.batch_size
    xs = x_test[s:e]
    ys = y_test[s:e]
    np_sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
    total_correct += np_sum_correct

acc = total_correct / 10000
print ("acc: %f" % (acc))
        
####################################




