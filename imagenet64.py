

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--noise', type=float, default=2.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--name', type=str, default="imagenet_weights")
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/datasets/imagenet64/tfrecord/val/'
    train_path = '/home/bcrafton3/Data_SSD/datasets/imagenet64/tfrecord/train/'
else:
    val_path = '/usr/scratch/datasets/imagenet64/tfrecord/val/'
    train_path = '/usr/scratch/datasets/imagenet64/tfrecord/train/'

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

##############################################

def get_val_filenames():
    val_filenames = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            val_filenames.append(os.path.join(val_path, file))

    np.random.shuffle(val_filenames)

    remainder = len(val_filenames) % args.batch_size
    if (remainder):
        val_filenames = val_filenames[:(-remainder)]

    return val_filenames

def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            train_filenames.append(os.path.join(train_path, file))

    np.random.shuffle(train_filenames)

    remainder = len(train_filenames) % args.batch_size
    if (remainder):
        train_filenames = train_filenames[:(-remainder)]

    return train_filenames

def extract_fn(record):
    _feature={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    sample = tf.parse_single_example(record, _feature)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    # this was tricky ... stored as uint8, not float32.
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (1, 64, 64, 3))

    # means = tf.reshape(tf.constant(MEAN), [1, 1, 1, 3])
    # image = (image - means) / 255. * 2.
    image, _ = quantize(image, 0, 127)
    

    label = sample['label']
    return [image, label]

###############################################################

train_filenames = get_train_filenames()
val_filenames = get_val_filenames()

filename = tf.placeholder(tf.string, shape=[None])

###############################################################

val_dataset = tf.data.TFRecordDataset(filename)
val_dataset = val_dataset.map(extract_fn, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_dataset = tf.data.TFRecordDataset(filename)
train_dataset = train_dataset.map(extract_fn, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (args.batch_size, 64, 64, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

def batch_norm(x, f, name):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32, name=name+'_gamma')
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32, name=name+'_beta')
    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p, name):
    filters1 = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv1')
    filters2 = tf.Variable(init_filters(size=[3,3,f2,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv2')

    conv1 = tf.nn.conv2d(x, filters1, [1,1,1,1], 'SAME')
    # bn1   = batch_norm(conv1, f2, name+'_bn1')
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.nn.conv2d(relu1, filters2, [1,1,1,1], 'SAME')
    # bn2   = batch_norm(conv2, f2, name+'_bn2')
    relu2 = tf.nn.relu(bn2)

    pool = tf.nn.avg_pool(relu2, ksize=[1,p,p,1], strides=[1,p,p,1], padding='SAME')

    return pool

###############################################################

'''
block1 = block(features, 3,  64,   2, 'block1')                                      # 64
block2 = block(block1,   64,  128,  2, 'block2')                                     # 32
block3 = block(block2,   128, 256,  2, 'block3')                                     # 16
block4 = block(block3,   256, 512,  2, 'block4')                                     # 8
block5 = block(block4,   512, 1024, 1, 'block5')                                     # 4
pool   = tf.nn.avg_pool(block5, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')  # 1

flat   = tf.reshape(pool, [args.batch_size, 1024])

mat1   = tf.Variable(init_matrix(size=(1024, 1000), init='alexnet'), dtype=tf.float32, name='fc1')
bias1  = tf.Variable(np.zeros(shape=1000), dtype=tf.float32, name='fc1_bias')
fc1    = tf.matmul(flat, mat1) + bias1
'''

'''
m = model(layers=[
conv_block(3,    64, 1, noise=args.noise),
conv_block(64,   64, 2, noise=args.noise),

conv_block(64,   128, 1, noise=args.noise),
conv_block(128,  128, 2, noise=args.noise),

conv_block(128,  256, 1, noise=args.noise),
conv_block(256,  256, 2, noise=args.noise),

conv_block(256,  512, 1, noise=args.noise),
conv_block(512,  512, 2, noise=args.noise),

conv_block(512,  1024, 1, noise=args.noise),
conv_block(1024, 1024, 1, noise=args.noise),

avg_pool(4, 4),
dense_block(1024, 1000, noise=args.noise)
])
'''

m = model(layers=[
conv_block(3,    64, 1, noise=args.noise),
conv_block(64,   64, 2, noise=args.noise),
conv_block(64,   128, 2, noise=args.noise),
conv_block(128,  256, 2, noise=args.noise),
conv_block(256,  512, 2, noise=args.noise),
conv_block(512,  1024, 1, noise=args.noise),

avg_pool(4, 4),
dense_block(1024, 1000, noise=args.noise)
])

###############################################################

learning_rate = tf.placeholder(tf.float32, shape=())
scale = tf.placeholder(tf.float32, [len(m.layers)])
model_train = m.train(x=features)
model_collect = m.collect(x=features)
model_predict = m.predict(x=features, scale=scale)

predict = tf.argmax(model_predict, axis=1)
correct = tf.equal(predict, tf.argmax(labels, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

###############################################################

weights = m.get_weights()

####################################

'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc1, labels=labels))
correct = tf.equal(tf.argmax(fc1, axis=1), tf.argmax(labels, 1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

train = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=args.eps).minimize(loss)
'''

loss_class = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=model_train))
params = tf.trainable_variables()

loss_l2 = []
for p in params:
    loss_l2.append(tf.nn.l2_loss(p))
loss_l2 = tf.reduce_sum(loss_l2)

beta = 0.001 
loss = loss_class + beta * loss_l2

###############################################################

grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=args.eps).apply_gradients(grads_and_vars)

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

for ii in range(0, args.epochs):
    print('epoch %d/%d' % (ii, args.epochs))

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    start = time.time()
    for jj in range(0, len(train_filenames), args.batch_size):
        sess.run([train], feed_dict={handle: train_handle, learning_rate: args.lr})
        if (jj % (100 * args.batch_size) == 0):
            img_per_sec = (jj + args.batch_size) / (time.time() - start)
            p = "%d | img/s: %f" % (jj, img_per_sec)
            print (p)

##################################################################

# probably want to use training data for this ...
sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})

scales = []

start = time.time()
for jj in range(0, len(val_filenames), args.batch_size):
    np_model_collect = sess.run(model_collect, feed_dict={handle: val_handle, learning_rate: 0.0})
    scales.append(np_model_collect)
    if (jj % (100 * args.batch_size) == 0):
        img_per_sec = (jj + args.batch_size) / (time.time() - start)
        p = "%d | img/s: %f" % (jj, img_per_sec)
        print (p)

scales = np.ceil(np.average(scales, axis=0))

##################################################################
            
sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})

total_correct = 0

for jj in range(0, len(val_filenames), args.batch_size):
    np_sum_correct = sess.run(sum_correct, feed_dict={handle: val_handle, scale: scales, learning_rate: 0.0})
    total_correct += np_sum_correct

acc = total_correct / len(val_filenames)
print ("acc: %f" % (acc))

##################################################################

weight_dict = sess.run(weights, feed_dict={})

for key in weight_dict.keys():
    (w, b) = weight_dict[key]
    weight_dict[key] = (w, b, scales[key])

weight_dict['acc'] = acc

np.save(args.name, weight_dict)

##################################################################

