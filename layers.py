
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x):
        y = x
        for layer in self.layers:
            y = layer.train(y)
        return y
    
    def collect(self, x):
        scale = {}
        y = x 
        for layer in self.layers:
            y, s = layer.collect(y)
            scale.update(s)
        return y, scale

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
        self.noise = noise
        
        self.relu = relu
        
        if weights:
            print (self.layer_id, shape, weights[self.layer_id].keys())
            f, g, b, mean, var = weights[self.layer_id]['f'], weights[self.layer_id]['g'], weights[self.layer_id]['b'], weights[self.layer_id]['mean'], weights[self.layer_id]['var']
            assert (np.shape(f) == shape)
            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.g = tf.Variable(g, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.mean = tf.Variable(mean, dtype=tf.float32, trainable=False)
            self.var = tf.Variable(var, dtype=tf.float32, trainable=False)
        else:
            self.f = tf.Variable(init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32)
            self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32)

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,self.p,self.p,1], 'SAME') 
        
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(self.var + 1e-5)
        bn = ((conv - self.mean) / std) * self.g + self.b
        
        # bn = tf.Print(bn, [tf.reduce_max(mean), tf.reduce_max(self.mean)], message='', summarize=1000)
        # bn = tf.Print(bn, [self.layer_id, tf.reduce_max(std), tf.reduce_max(var), tf.reduce_max(self.var)], message='', summarize=1000)
                
        if self.relu:
            out = tf.nn.relu(bn)
        else:
            out = bn
        
        return out
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)
        
#############

class res_block1(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1

        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise

        if weights:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
        else:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=None)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=None, relu=False)

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(y2 + x)
        return y3

    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)

#############

class res_block2(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1

        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise

        if weights:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
            self.conv3 = conv_block((1, 1, f1, f2), p, noise=None, weights=weights, relu=False)
        else:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=None)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=None, relu=False)
            self.conv3 = conv_block((1, 1, f1, f2), p, noise=None, weights=None, relu=False)

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = self.conv3.train(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4

    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)

#############

class dense_block(layer):
    def __init__(self, isize, osize, noise, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.isize = isize
        self.osize = osize
        self.noise = noise
        
        if weights:
            w, b = weights[self.layer_id]['w'], weights[self.layer_id]['b']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
        else:
            self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
        
    def train(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def get_weights(self):
        assert (False)








        
        
        
