
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

def quantize(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    return x, scale

def quantize_and_dequantize(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    x = x * scale
    return x, scale
    
#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        y = x
        for layer in self.layers:
            y = layer.predict(y)
        return y
        
    def qpredict(self, x):
        y = x
        for layer in self.layers:
            y = layer.qpredict(y)
        return y
        
    def get_weights(self):
        weights_dict = {}
        for layer in self.layers:
            weights_dict.update(layer.get_weights())
        return weights_dict
        
#############

class layer:
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)
    
    def collect(self, x):
        assert(False)

    def predict(self, x):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, noise, weights=None, relu=True):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        self.relu = relu
        
        if 'g' in weights[self.layer_id].keys():
            f, b, g, mean, var = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['g'], weights[self.layer_id]['mean'], weights[self.layer_id]['var']
            var = np.sqrt(var + 1e-5)
            f = f * (g / var)
            b = b - (g / var) * mean
            self.f = tf.Variable(f, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
        else:
            f, b = weights[self.layer_id]['f'], weights[self.layer_id]['b']
            self.f = tf.Variable(f, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
            
        assert (np.shape(f) == shape)
        self.x_max1 = 0
        self.y_max1 = 0
        self.x_max2 = 0
        self.y_max2 = 0

    def predict(self, x):    
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        self.x_max1 = tf.maximum(tf.reduce_max(tf.abs(x_pad)), self.x_max1)
        
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + self.b

        if self.relu:
            out = tf.nn.relu(conv)
        else:
            out = conv

        self.y_max1 = tf.maximum(tf.reduce_max(tf.abs(out)), self.y_max1)
        return out

    def qpredict(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        self.x_max2 = tf.maximum(tf.reduce_max(tf.abs(x_pad)), self.x_max2)
        
        qf, f_scale = quantize(self.f, -128, 127)
        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + self.b * self.x_max2 / self.x_max1 * f_scale

        if self.relu:
            out = tf.nn.relu(conv)
        else:
            out = conv

        self.y_max2 = tf.maximum(tf.reduce_max(tf.abs(out)), self.y_max2)
        
        qout, out_scale = quantize(out, -128, 127)
        return qout
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict
        
#############

class res_block1(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
        y3 = tf.nn.relu(y2 + x)
        return y3
        
    def qpredict(self, x):
        y1 = self.conv1.qpredict(x)
        y2 = self.conv2.qpredict(y1)
        y3 = tf.nn.relu(y2 + x)
        return y3

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict[self.layer_id] = {}
        return weights_dict
        
#############

class res_block2(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, noise=None, weights=weights, relu=False)

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
        y3 = self.conv3.predict(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4

    def qpredict(self, x):
        y1 = self.conv1.qpredict(x)
        y2 = self.conv2.qpredict(y1)
        y3 = self.conv3.qpredict(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4
        
    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        weights3 = self.conv3.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)
        weights_dict[self.layer_id] = {}
        return weights_dict

#############

class dense_block(layer):
    def __init__(self, isize, osize, noise, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.isize = isize
        self.osize = osize
        
        w, b = weights[self.layer_id]['w'], weights[self.layer_id]['b']
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def predict(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc
        
    def qpredict(self, x):
        return self.predict(x)
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'w': self.w, 'b': self.b}
        return weights_dict

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1
    
        self.s = s
        self.p = p

    def predict(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool

    def qpredict(self, x):
        return self.predict(x)
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def predict(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
        
    def qpredict(self, x):
        return self.predict(x)
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict








        
        
        
