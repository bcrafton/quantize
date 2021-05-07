
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
# x_train = quantize_np(x_train, -128, 127)
x_train = x_train // 2

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = quantize_np(x_test, -128, 127)
x_test = x_test // 2

####################################

model = model(layers=[
conv_block((3,3, 3,32), 1, weights=weights, train=False),
conv_block((3,3,32,32), 2, weights=weights, train=False),
conv_block((3,3,32,32), 1, weights=weights, train=False),
conv_block((3,3,32,32), 1, weights=weights, train=False),
conv_block((3,3,32,32), 2, weights=weights, train=False),
dense_block(32*4*4, 10, weights=weights, train=False)
])

####################################

def predict(model, x, y):
    pred_logits, all_ys = model.predict(x)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct, all_ys

####################################

batch_size = 50
total_correct = 0
for batch in range(0, len(x_test), batch_size):
    xs = x_test[batch:batch+batch_size].astype(np.float32)
    ys = y_test[batch:batch+batch_size].reshape(-1).astype(np.int64)

    correct, all_ys = predict(model, xs, ys)
    total_correct += correct

    if (batch == 0):
        np.savetxt("y1_ref", X=all_ys[0][2].transpose(2,0,1).flatten(), fmt="%d")
        np.savetxt("y2_ref", X=all_ys[1][2].transpose(2,0,1).flatten(), fmt="%d")
        np.savetxt("y3_ref", X=all_ys[2][2].transpose(2,0,1).flatten(), fmt="%d")
        np.savetxt("y4_ref", X=all_ys[3][2].transpose(2,0,1).flatten(), fmt="%d")
        np.savetxt("y5_ref", X=all_ys[4][2].transpose(2,0,1).flatten(), fmt="%d")
        np.savetxt("y6_ref", X=all_ys[5][2].flatten(), fmt="%d")

print (total_correct / len(x_test) * 100)

####################################














