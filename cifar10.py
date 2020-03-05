
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

####################################

import numpy as np
import tensorflow as tf
import keras

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

epochs = 10
batch_size = 50
x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

####################################

def block(x, f1, f2, p, name):
    f = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv')
    qf = tf.quantization.quantize_and_dequantize(input=f, input_min=-127, input_max=127, signed_input=True, num_bits=8)
    
    conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME')
    relu = tf.nn.relu(conv)
    pool = tf.nn.avg_pool(relu, ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')

    qpool = tf.quantization.quantize_and_dequantize(input=pool, input_min=-127, input_max=127, signed_input=True, num_bits=8)
    return qpool

def dense(x, size, name):    
    w = tf.Variable(init_matrix(size=size, init='alexnet'), dtype=tf.float32, name=name)
    qw = tf.quantization.quantize_and_dequantize(input=w, input_min=-127, input_max=127, signed_input=True, num_bits=8)
    
    fc = tf.matmul(x, qw)
    
    qfc = tf.quantization.quantize_and_dequantize(input=fc, input_min=-127, input_max=127, signed_input=True, num_bits=8)
    return qfc

####################################

block1 = block(x,       3, 32,  2, 'block1') # 32 -> 16
block2 = block(block1, 32, 64,  2, 'block2') # 16 -> 8
block3 = block(block2, 64, 128, 2, 'block3') #  8 -> 4
pool   = tf.nn.avg_pool(block3, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')  # 4 -> 1
flat   = tf.reshape(pool, [batch_size, 128])
out    = dense(flat, [128, 10], 'fc1')

####################################

predict = tf.argmax(out, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

####################################

for ii in range(epochs):
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0

    for jj in range(0, 10000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
  
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
####################################








