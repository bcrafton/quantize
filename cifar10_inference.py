
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50)
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
  
####################

def merge_dicts(list_of_dicts):
    results = {}
    for d in list_of_dicts:
        for key in d.keys():
            if key in results.keys():
                results[key].append(d[key])
            else:
                results[key] = [d[key]]

    return results
  
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
conv_block(3,   64, 1, weights=weights[0]),
conv_block(64,  64, 2, weights=weights[1]),

conv_block(64,  128, 1, weights=weights[2]),
conv_block(128, 128, 2, weights=weights[3]),

conv_block(128, 256, 1, weights=weights[4]),
conv_block(256, 256, 2, weights=weights[5]),

avg_pool(4, 4, weights=weights[6]),
dense_block(256, 10, weights=weights[7])
])

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
# noise = tf.placeholder(tf.float32, [len(m.layers)])
noise = tf.placeholder(tf.float32, [7])

model_predict = m.inference(x=x, noise=noise)

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

num_layers = 7
results = np.load('results.npy', allow_pickle=True).item()

var = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15])

y_std = np.zeros(shape=(2, 2, len(var), num_layers))
acc = np.zeros(shape=(2, 2, len(var)))

for key in sorted(results.keys()):
    (skip, cards, sigma) = key
    layer_results = results[key]

    for layer in range(num_layers):
        example_results = merge_dicts(layer_results[layer])
        sigma_index = np.where(var == sigma)[0][0]
        y_std[skip][cards][sigma_index][layer] = np.mean(example_results['std'])
        acc[skip][cards][sigma_index] = layer_results['acc']

####################################

y_std = np.reshape(y_std, (2 * 2 * len(var), num_layers))

for std in y_std:
    total_correct = 0
    for jj in range(0, 10000, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        np_sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys, noise: std / 0.55 })
        total_correct += np_sum_correct

    acc = total_correct / 10000
    # print (std)
    print ("acc: %f" % (acc))
        
####################################




