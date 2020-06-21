
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

####################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

####################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
# x_train = quantize_np(x_train, 0, 127)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
# x_test = quantize_np(x_test, 0, 127)

####################################

class Conv(tf.keras.layers.Layer):
    def __init__(self, k, f):
        super(Conv, self).__init__()
        self.k = k
        self.f = f

    def build(self, input_shape):
        _, _, _, c = input_shape
        self.kernel = self.add_weight("kernel", shape=[self.k, self.k, c, self.f])
        self.bias = self.add_weight("kernel", shape=[self.f])

    def call(self, input):
        # return tf.matmul(input, self.kernel)
        return tf.nn.relu(tf.nn.conv2d(input, self.kernel, 1, 'SAME') + self.bias)

####################################

model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv(3, 32))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Conv(3, 64))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Conv(3, 64))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

####################################

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

####################################




