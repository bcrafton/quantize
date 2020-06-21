
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

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

class Conv(tf.keras.layers.Layer):
    def __init__(self, k, f, w, b):
        super(Conv, self).__init__()
        self.k = k
        self.f = f
        self.w = tf.Variable(w, trainable=True)

    def build(self, input_shape):
        _, _, _, c = input_shape
        self.g = tf.Variable(np.ones(shape=self.f).astype(np.float32), trainable=True)
        self.b = tf.Variable(np.ones(shape=self.f).astype(np.float32), trainable=True)
        
    def call(self, input, training=True):
        if training:
            conv = tf.nn.conv2d(input, self.w, 1, 'SAME')
            mean, var = tf.nn.moments(conv, axes=[0,1,2])
            std = tf.sqrt(var + 1e-5)
            fold_w = (self.g * self.w) / std
            fold_b = self.b - ((self.g * mean) / std)
            qw = quantize_and_dequantize(fold_w, -128, 127)
            
            y = tf.nn.conv2d(input, qw, [1,1,1,1], 'SAME') + fold_b
            a = tf.nn.relu(y)
            q = quantize_and_dequantize(a, -128, 127)
            return q
        else:
            conv = tf.nn.conv2d(input, self.w, 1, 'SAME')
            mean, var = tf.nn.moments(conv, axes=[0,1,2])
            std = tf.sqrt(var + 1e-5)
            fold_w = (self.g * self.w) / std
            fold_b = self.b - ((self.g * mean) / std)
            qw, scale = quantize(fold_w, -128, 127)
            qb = quantize_predict(fold_b, scale, -2**24, 2**24-1)
            
            y = tf.nn.conv2d(input, qw, [1,1,1,1], 'SAME') + qb
            a = tf.nn.relu(y)
            q, scale = quantize(a, -128, 127)
            return q

####################################

w1 = np.random.normal(loc=0., scale=0.05, size=(3,3,3,32)).astype(np.float32)
b1 = np.zeros(shape=32).astype(np.float32)
w2 = np.random.normal(loc=0., scale=0.05, size=(3,3,32,64)).astype(np.float32)
b2 = np.zeros(shape=64).astype(np.float32)
w3 = np.random.normal(loc=0., scale=0.05, size=(3,3,64,64)).astype(np.float32)
b3 = np.zeros(shape=64).astype(np.float32)

####################################

model = models.Sequential()
model.add(Conv(3, 32, w1, b1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv(3, 64, w2, b2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Conv(3, 64, w3, b3))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

####################################

optimizer = tf.keras.optimizers.Adam()

def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model(x, training=True)
        pred_label = tf.argmax(pred_logits, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    
    grad = tape.gradient(loss, model.trainable_variables)
    return loss, correct, grad

####################################

def predict(model, x, y):
    pred_logits = model(x, training=False)
    pred_label = tf.argmax(pred_logits, axis=1)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
    return correct

####################################

batch_size = 50
for epoch in range(5):
    total_correct = 0
    for batch in range(0, len(x_train), batch_size):
        xs = x_train[batch:batch+batch_size]
        ys = y_train[batch:batch+batch_size].reshape(-1).astype(np.int32)
        
        loss, correct, grad = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

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



















