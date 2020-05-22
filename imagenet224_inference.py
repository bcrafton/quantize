
import argparse
import os
import sys
import time

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='imagenet224')
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/val/'
    train_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/train/'
else:
    val_path = '/usr/scratch/datasets/imagenet224/val/'
    train_path = '/usr/scratch/datasets/imagenet224/train/'

val_labels = './imagenet_labels/validation_labels.txt'
train_labels = './imagenet_labels/train_labels.txt'

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

##############################################

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)
    '''
    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    '''
    return image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def train_preprocess(image, label):
    '''
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)
    '''
    # image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)  
    # image = tf.Print(image, [tf.shape(image)], message='', summarize=1000)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.central_crop(image, 0.875)
    
    image = image / 255.
    # image = image - tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, 3])
    image = image / tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, 3])

    scale = 127 / tf.reduce_max(tf.abs(image))
    image = image * scale
    image = tf.round(image)
    image = tf.clip_by_value(image, 0, 127)

    return image, label
    

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    '''
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)
    '''
    # image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    
    # image = tf.Print(image, [tf.shape(image)], message='', summarize=1000)
    image = tf.image.resize(image, (256, 256))
    image = tf.image.central_crop(image, 0.875)

    image = image / 255.
    # image = image - tf.reshape(tf.constant([0.485, 0.456, 0.406]), [1, 1, 3])
    image = image / tf.reshape(tf.constant([0.229, 0.224, 0.225]), [1, 1, 3])

    scale = 127 / tf.reduce_max(tf.abs(image))
    image = image * scale
    image = tf.round(image)
    image = tf.clip_by_value(image, 0, 127)
    # image = tf.Print(image, [tf.reduce_max(image), tf.reduce_min(image)], message='', summarize=1000)

    return image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            validation_images.append(os.path.join(val_path, file))
    validation_images = sorted(validation_images)

    validation_labels_file = open(val_labels)
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    remainder = len(validation_labels) % args.batch_size
    if remainder:
        validation_images = validation_images[:(-remainder)]
        validation_labels = validation_labels[:(-remainder)]

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    f = open(train_labels, 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building train dataset")

    for subdir, dirs, files in os.walk(train_path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
    if remainder:
        training_images = training_images[:(-remainder)]
        training_labels = training_labels[:(-remainder)]

    return training_images, training_labels

###############################################################

filename = tf.placeholder(tf.string, shape=[None])
label = tf.placeholder(tf.int64, shape=[None])

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
# val_dataset = val_dataset.shuffle(len(val_imgs))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(16)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(16)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

weights = np.load('resnet18_quant_weights.npy', allow_pickle=True).item()

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

model_predict = m.predict(x=features)

predict = tf.argmax(model_predict, axis=1)
correct = tf.equal(predict, tf.argmax(labels, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})

total_correct = 0
start = time.time()
for jj in range(0, len(val_imgs), args.batch_size):
    [np_sum_correct] = sess.run([sum_correct], feed_dict={handle: val_handle})
    total_correct += np_sum_correct
    if (jj % (100 * args.batch_size) == 0):
        acc = total_correct / (jj + args.batch_size)
        img_per_sec = (jj + args.batch_size) / (time.time() - start)
        p = "%d | acc: %f | img/s: %f" % (jj, acc, img_per_sec)
        print (p)

##################################################################








