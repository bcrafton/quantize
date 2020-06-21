
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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = quantize_np(x_train, 0, 127)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = quantize_np(x_test, 0, 127)

####################################

model = model(layers=[
conv_block(3,   64, 1),
conv_block(64,  64, 2),

conv_block(64,  128, 1),
conv_block(128, 128, 2),

conv_block(128, 256, 1),
conv_block(256, 256, 2),

avg_pool(4, 4),
dense_block(256, 10)
])

params = model.get_params()

####################################

optimizer = tf.keras.optimizers.Adam()

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

batch_size = 50
for epoch in range(5):
    total_correct = 0
    for batch in range(0, len(x_train), batch_size):
        print (batch)
    
        xs = x_train[batch:batch+batch_size]
        ys = y_train[batch:batch+batch_size].reshape(-1).astype(np.int32)
        
        loss, correct, grad = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, params))

        total_correct += correct
        
    print (total_correct / len(x_train) * 100)

####################################

model.save('./results')

####################################

batch_size = 50
total_correct = 0
for batch in range(0, len(x_train), batch_size):
    xs = x_train[batch:batch+batch_size]
    ys = y_train[batch:batch+batch_size].reshape(-1).astype(np.int32)
    
    correct = predict(model, xs, ys)
    total_correct += correct
    
print (total_correct / len(x_train) * 100)

####################################



















