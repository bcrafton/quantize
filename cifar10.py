
import os
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

epochs = 20
batch_size = 50

m = model(layers=[
conv_block(3,   32, 1),
conv_block(32,  32, 2),

conv_block(32,  64, 1),
conv_block(64,  64, 2),

conv_block(64,  128, 1),
conv_block(128, 128, 2),

avg_pool(4, 4),
dense_block(128, 10)
])

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
scale = tf.placeholder(tf.float32, [len(m.layers)])

model_train = m.train(x=x)
model_collect = m.collect(x=x)
model_predict = m.predict(x=x, scale=scale)

####################################

weights = m.get_weights()

####################################

predict = tf.argmax(model_predict, axis=1)
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

'''
loss_l1 = []
for p in params:
    loss_l1.append(tf.reduce_sum(tf.abs(p)))
loss_l1 = tf.reduce_sum(loss_l1)
loss = loss_class + 0.0001 * loss_l1
'''

####################################

'''
loss_exp = []
for p in params:
    loss_exp.append(tf.reduce_sum(tf.exp(-1. * tf.abs(p) / tf.reduce_max(tf.abs(p)))))
loss_exp = tf.reduce_sum(loss_exp)
loss = loss_class + 0.00001 * loss_exp
'''

####################################
# this dont work it seems.
# also would need to offset x.
# we never add 128.
'''
def count_ones(x):
    count = 0
    for bit in range(8):
        count += np.bitwise_and(np.right_shift(x, bit), 1)
    return count

cost_table_np = np.zeros(shape=256)
for ii in range(256):
    cost_table_np[ii] = count_ones(ii)
    
cost_table = tf.constant(cost_table_np, dtype=tf.int32)

loss_bit = []
for p in params:
    qp, _ = quantize(p, -128, 127)
    qp_flat = tf.reshape(qp, [-1])
    qp_flat_int = tf.cast(qp_flat, dtype=tf.int32)
    gather = tf.gather(params=cost_table, indices=qp_flat_int)
    loss_bit_p = tf.cast(tf.reduce_sum(gather), dtype=tf.float32)
    loss_bit.append(tf.reduce_sum(loss_bit_p))
    
loss_bit = tf.reduce_sum(loss_bit)
loss = loss_class + 0.0001 * loss_bit
'''
####################################
'''
# l1 loss with higher cost on 
# 1) small negative numbers
# 2) large positive numbers.
'''
####################################

grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):
    print ("epoch %d/%d" % (ii, epochs))
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
####################################

scales = []
for jj in range(0, 50000, batch_size):
    s = jj
    e = jj + batch_size
    xs = x_train[s:e]
    ys = y_train[s:e]
    np_model_collect = sess.run(model_collect, feed_dict={x: xs, y: ys})
    scales.append(np_model_collect)
    
# this needs to be ceil for cases where (scale < 1), like avg_pool.
scales = np.ceil(np.average(scales, axis=0))
    
####################################

total_correct = 0
for jj in range(0, 10000, batch_size):
    s = jj
    e = jj + batch_size
    xs = x_test[s:e]
    ys = y_test[s:e]
    np_sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys, scale: scales})
    total_correct += np_sum_correct
        
print ("acc: " + str(total_correct * 1.0 / 10000))
        
####################################

weight_dict = sess.run(weights, feed_dict={})

for key in weight_dict.keys():
    (w, b) = weight_dict[key]
    weight_dict[key] = (w, b, scales[key])

np.save("cifar10_weights", weight_dict)

####################################




