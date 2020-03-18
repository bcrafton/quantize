
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
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

def create_stats_dict(stats_list):
    stats_dict = {}
    for example_dict in stats_list:
        for layer in example_dict.keys():
            layer_dict = example_dict[layer]
            for stat in layer_dict.keys():
                if layer not in stats_dict.keys():
                    stats_dict[layer] = {}

                stat_value = layer_dict[stat]
                if stat not in stats_dict[layer].keys():
                    stats_dict[layer][stat] = [stat_value]
                else:
                    stats_dict[layer][stat].append(stat_value)

    for layer in stats_dict.keys():
        for stat in stats_dict[layer].keys():
            if stat == 'var':
                stats_dict[layer][stat] = np.mean(np.sqrt(np.array(stats_dict[layer][stat]) + 1e-3), axis=0)
            else:
                stats_dict[layer][stat] = np.mean(stats_dict[layer][stat], axis=0)

    return stats_dict

##############################################

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

m = model(layers=[
conv_block(3,   32, 1, args.noise),
conv_block(32,  32, 2, args.noise),

conv_block(32,  64, 1, args.noise),
conv_block(64,  64, 2, args.noise),

conv_block(64,  128, 1, args.noise),
conv_block(128, 128, 2, args.noise),

avg_pool(4, 4),
dense_block(128, 10, args.noise)
])

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
scale = tf.placeholder(tf.float32, [len(m.layers)])

model_train = m.train(x=x)

####################################

weights = m.get_weights()

####################################

predict = tf.argmax(model_train, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

####################################

loss_class = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model_train))

params = tf.trainable_variables()

####################################

loss_l2 = []
for p in params:
    loss_l2.append(tf.nn.l2_loss(p))
loss_l2 = tf.reduce_sum(loss_l2)

# beta = 0.0   # 63%
# beta = 0.01  # 10%
beta = 0.001 # 70%
# beta = 0.003 # 67%
loss = loss_class + beta * loss_l2

####################################

grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=args.eps).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(args.epochs):
    print ("epoch %d/%d" % (ii, args.epochs))
    for jj in range(0, 50000, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
    weight_dict = sess.run(weights, feed_dict={})

    for key in weight_dict.keys():
        if key in np_stats_dict.keys():
            (w, g, b) = weight_dict[key]
            w = (g * w) / np_stats_dict[key]['var']
            b = b - ((g * np_stats_dict[key]['mean']) / np_stats_dict[key]['var'])
            weight_dict[key] = (w, b)

    weight_dict['acc'] = acc
    np.save(args.name, weight_dict)




