
import numpy as np
import tensorflow as tf
from layers import *
from load import Loader
import time

####################################

'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
'''

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu = gpus[1]
tf.config.experimental.set_visible_devices(gpu, 'GPU')
tf.config.experimental.set_memory_growth(gpu, True)

####################################

def quantize_np(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.floor(x)
    x = np.clip(x, low, high)
    return x

####################################

train_flag = True
num_example = 100000

if train_flag:
    weights = np.load('resnet18.npy', allow_pickle=True).item()
else:
    weights = np.load('trained_weights.npy', allow_pickle=True).item()

####################################

model = model(layers=[
conv_block((7,7,3,64), 2, weights=weights, train=train_flag),

max_pool(2, 3),

res_block1(64,   64, 1, weights=weights, train=train_flag),
res_block1(64,   64, 1, weights=weights, train=train_flag),

res_block2(64,   128, 2, weights=weights, train=train_flag),
res_block1(128,  128, 1, weights=weights, train=train_flag),

res_block2(128,  256, 2, weights=weights, train=train_flag),
res_block1(256,  256, 1, weights=weights, train=train_flag),

res_block2(256,  512, 2, weights=weights, train=train_flag),
res_block1(512,  512, 1, weights=weights, train=train_flag),

avg_pool(7, 7),
dense_block(512, 1000, weights=weights, train=train_flag)
])

if train_flag:
    params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1.)

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    
    grad = tape.gradient(loss, params)
    return loss, correct, grad

####################################

def predict(model, x, y):
    pred_logits = model.predict(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

def collect(model, x, y):
    pred_logits = model.collect(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

def run_train():

    # total = 1281150
    total = num_example
    total_correct = 0
    total_loss = 0
    batch_size = 50

    load = Loader('/home/bcrafton3/Data_HDD/keras_imagenet/keras_imagenet_train/', total // batch_size, batch_size, 8)
    start = time.time()

    for batch in range(0, total, batch_size):
        while load.empty(): pass # print ('empty')
        
        x, y = load.pop()
        
        loss, correct, grad = gradients(model, x, y)
        optimizer.apply_gradients(zip(grad, params))
        total_loss += loss.numpy()
        
        total_correct += correct.numpy()
        acc = round(total_correct / (batch + batch_size), 3)
        avg_loss = total_loss / (batch + batch_size)
        
        if (batch + batch_size) % (batch_size * 100) == 0:
            img_per_sec = (batch + batch_size) / (time.time() - start)
            print (batch + batch_size, img_per_sec, acc, avg_loss)

    load.join()
    # trained_weights = model.get_weights()
    # np.save('trained_weights', trained_weights)

####################################

def run_collect():

    # total = 1281150
    total = num_example
    total_correct = 0
    total_loss = 0
    batch_size = 50

    load = Loader('/home/bcrafton3/Data_HDD/keras_imagenet/keras_imagenet_train/', total // batch_size, batch_size, 8)
    start = time.time()

    for batch in range(0, total, batch_size):
        while load.empty(): pass # print ('empty')
        
        x, y = load.pop()
        correct = collect(model, x, y)

        total_correct += correct.numpy()
        acc = round(total_correct / (batch + batch_size), 3)
        avg_loss = total_loss / (batch + batch_size)
        
        if (batch + batch_size) % (batch_size * 100) == 0:
            img_per_sec = (batch + batch_size) / (time.time() - start)
            print (batch + batch_size, img_per_sec, acc, avg_loss)

    load.join()
    trained_weights = model.get_weights()
    np.save('trained_weights', trained_weights)

####################################

def run_val():

    total = 50000
    total_correct = 0
    total_loss = 0
    batch_size = 50

    load = Loader('/home/bcrafton3/Data_HDD/keras_imagenet/keras_imagenet_val/', total // batch_size, batch_size, 8)
    start = time.time()

    for batch in range(0, total, batch_size):
        while load.empty(): pass # print ('empty')
        
        x, y = load.pop()
        correct = predict(model, x, y)

        total_correct += correct.numpy()
        acc = round(total_correct / (batch + batch_size), 3)
        avg_loss = total_loss / (batch + batch_size)
        
        if (batch + batch_size) % (batch_size * 100) == 0:
            img_per_sec = (batch + batch_size) / (time.time() - start)
            print (batch + batch_size, img_per_sec, acc, avg_loss)

    load.join()

####################################

if train_flag:
    run_train()
    run_collect()
else:
    run_val()

####################################














