
import numpy as np
import tensorflow as tf
from layers import *

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

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = quantize_np(x_train, 0, 127)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = quantize_np(x_test, 0, 127)

####################################

model = model(layers=[
conv_block(3,   64, weights=weights),
conv_block(64,  64, weights=weights),
avg_pool(2, 2),

conv_block(64,  128, weights=weights),
conv_block(128, 128, weights=weights),
avg_pool(2, 2),

conv_block(128, 256, weights=weights),
conv_block(256, 256, weights=weights),
avg_pool(2, 2),

avg_pool(4, 4),
dense_block(256, 10, weights=weights)
])

####################################

def predict(model, x, y):
    pred_logits = model.predict(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

batch_size = 50
total_correct = 0
for batch in range(0, len(x_test), batch_size):
    xs = x_test[batch:batch+batch_size].astype(np.float32)
    ys = y_test[batch:batch+batch_size].reshape(-1).astype(np.int64)

    correct = predict(model, xs, ys)
    total_correct += correct

print (total_correct / len(x_test) * 100)

####################################














