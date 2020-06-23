
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
gpu = gpus[0]
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

weights = np.load('resnet18.npy', allow_pickle=True).item()

####################################

model = model(layers=[
conv_block((7,7,3,64), 2, weights=weights),

max_pool(2, 3),

res_block1(64,   64, 1, weights=weights),
res_block1(64,   64, 1, weights=weights),

res_block2(64,   128, 2, weights=weights),
res_block1(128,  128, 1, weights=weights),

res_block2(128,  256, 2, weights=weights),
res_block1(256,  256, 1, weights=weights),

res_block2(256,  512, 2, weights=weights),
res_block1(512,  512, 1, weights=weights),

avg_pool(7, 7),
dense_block(512, 1000, weights=weights)
])

params = model.get_params()

####################################

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1.)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1.)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x, training=True)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits))
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    
    grad = tape.gradient(loss, params)
    return loss, correct, grad

####################################

def predict(model, x, y):
    pred_logits = model.train(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

def run_train():

    # total = 1281150
    total = 35000
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

run_val()
run_train()
run_val()

















