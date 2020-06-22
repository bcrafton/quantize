
import numpy as np
import tensorflow as tf
from layers import *
from load import Loader
import time

####################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

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

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
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

total = 50000
total_correct = 0
total_loss = 0
batch_size = 32

for epoch in range(2):
    load = Loader('/home/brian/Desktop/ILSVRC2012/val')
    start = time.time()
    
    for batch in range(0, total, batch_size):
        while load.empty(): pass
        
        x, y = load.pop()
        
        # if inference:
        # correct = predict(model, x, y)
        # if train:
        loss, correct, grad = gradients(model, x, y)
        optimizer.apply_gradients(zip(grad, params))
        
        total_correct += correct.numpy()
        total_loss += np.sum(loss.numpy())
        
        acc = round(total_correct / (batch + batch_size), 3)
        avg_loss = total_loss / (batch + batch_size)

        print (epoch, batch, acc, avg_loss)

    load.join()

####################################























