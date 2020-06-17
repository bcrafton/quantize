
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

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
        
        f, b = weights[self.layer_id]['f'], weights[self.layer_id]['b']
        assert (np.shape(f) == shape)
        self.f = tf.Variable(f, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)
        g, mean, var = weights[self.layer_id]['g'], weights[self.layer_id]['mean'], weights[self.layer_id]['var']
        self.g = tf.Variable(g, dtype=tf.float32)
        self.mean = tf.Variable(mean, dtype=tf.float32, trainable=False)
        self.var = tf.Variable(var, dtype=tf.float32, trainable=False)

    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
        mean, var = tf.nn.moments(conv, axes=[0,1,2])
        bn = tf.nn.batch_normalization(conv, self.mean, self.var, self.b, self.g, 1e-5)

        if self.relu:
            out = tf.nn.relu(bn)
        else:
            out = bn
            
        return out
        
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

    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
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

    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
        y3 = self.conv3.predict(x)
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
        
    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc
        
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
        
    def train(self, x):        
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
        
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
        
    def train(self, x):        
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict








        
        
        
