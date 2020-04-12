
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

# TODO: quantize followed by dequantize does not work.

def quantize_and_dequantize(x, low, high):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Floor": "Identity"}):
        scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
        x = x / scale
        x = tf.floor(x)
        x = tf.clip_by_value(x, low, high)
        x = x * scale
        return x

def quantize(x, low, high):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Floor": "Identity"}):
        scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
        x = x / scale
        x = tf.floor(x)
        x = tf.clip_by_value(x, low, high)
        return x, scale
        
def dequantize(x, low, high):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Floor": "Identity"}):
        scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
        x = x * scale
        return x
    
def quantize_predict(x, scale, low, high):
    x = x / scale
    x = tf.floor(x)
    x = tf.clip_by_value(x, low, high)
    return x

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
        self.pad = self.k // 2
        
        self.noise = noise

        self.relu = relu
        
        if weights:
            f, b, g = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['g']
            assert (np.shape(f) == shape)
            print (self.layer_id, np.shape(f))
            self.f = tf.Variable(f, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
            self.g = tf.Variable(g, dtype=tf.float32)
        else:
            self.f = tf.Variable(init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32)
            self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32)
            self.q = None

    def train(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])

        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') # there is no bias when we have bn.
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-5)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        qf = quantize_and_dequantize(fold_f, -128, 127)
        # qb = quantize_and_dequantize(fold_b, -128, 127) 
        
        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + fold_b

        if self.relu:
            out = tf.nn.relu(conv)
        else:
            out = conv

        qout = quantize_and_dequantize(out, -128, 127)
        return qout
    
    def collect(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME') # there is no bias when we have bn.
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        # qf = quantize_and_dequantize(fold_f, -128, 127)
        qf, sf = quantize(fold_f, -128, 127)
        qb = quantize_predict(fold_b, sf, -2**24, 2**24-1)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')

        qpool, spool = quantize(pool, -128, 127)

        # qpool = tf.Print(qpool, [tf.reduce_max(pool), tf.reduce_mean(qpool)], message='', summarize=1000)
        # qpool = tf.Print(qpool, [self.layer_id, spool, tf.math.reduce_std(qf), tf.math.reduce_std(qb)], message='', summarize=1000)
        # return qpool, [spool, std, mean]
        return qpool, {self.layer_id: {'scale': spool, 'std': std, 'mean': mean}}

    def predict(self, x):
        qf, sf = quantize(self.f, -128, 127)
        qb = quantize_predict(self.b, sf, -2**24, 2**24-1)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        
        qpool = quantize_predict(pool, self.q, -128, 127)

        # qpool = tf.Print(qpool, [self.layer_id, scale, tf.math.reduce_std(qf), tf.math.reduce_std(qb)], message='', summarize=1000)
        return qpool
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict
        
#############

class res_block1(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise
        
        if weights:
            # self.q = weights[self.layer_id]['q']
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
        else:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=None)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=None, relu=False)

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(y2 + x)
        y3 = quantize_and_dequantize(y3, -128, 127)
        return y3
    
    def collect(self, x):
        y1, params1 = self.conv1.collect(x)
        y2, params2 = self.conv2.collect(y1)
        y3, s3 = quantize(y2 + x, -128, 127)
        params3 = {self.layer_id: {'scale': s3}}
        
        params = {}
        params.update(params1)
        params.update(params2)
        params.update(params3)
        return y3, params

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
        y3 = quantize_predict(y2 + x, self.q, -128, 127)
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
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise
        
        if weights:
            # self.q = weights[self.layer_id]['q']
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
        y4 = quantize_and_dequantize(y4, -128, 127)
        return y4
    
    def collect(self, x):
        y1, params1 = self.conv1.collect(x)
        y2, params2 = self.conv2.collect(y1)
        y3, params3 = self.conv3.collect(x)
        y4, s4 = quantize(y2 + y3, -128, 127)
        params4 = {self.layer_id: {'scale': s4}}
        
        params = {}
        params.update(params1)
        params.update(params2)
        params.update(params3)
        params.update(params4)
        return y4, params

    def predict(self, x):
        y1 = self.conv1.predict(x)
        y2 = self.conv2.predict(y1)
        y3 = self.conv3.predict(x)
        y4 = quantize_predict(y2 + y3, self.q, -128, 127)
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
        self.noise = noise
        
        if weights:
            w, b = weights[self.layer_id]['w'], weights[self.layer_id]['b']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            # self.q = q
        else:
            self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
            self.q = None
        
    def train(self, x):
        qw = quantize_and_dequantize(self.w, -128, 127)
        # qb = quantize_and_dequantize(self.b, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + self.b
        qfc = quantize_and_dequantize(fc, -128, 127)
        return qfc
    
    def collect(self, x):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) # + qb
        qfc, sfc = quantize(fc, -128, 127)
        return qfc, {self.layer_id: {'scale': sfc}}

    def predict(self, x):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) # + qb
        qfc = quantize_predict(fc, self.q, -128, 127)
        return qfc
        
    def get_weights(self):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
    
        weights_dict = {}
        weights_dict[self.layer_id] = {'w': qw, 'b': qb}
        
        return weights_dict

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
        if weights:
            self.q = weights[self.layer_id]['q']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_and_dequantize(pool, -128, 127)
        return qpool
    
    def collect(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return qpool, {self.layer_id: {'scale': spool}}

    def predict(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = quantize_predict(pool, self.q, -128, 127) # this only works because we use np.ceil(scales)
        return qpool
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
        if weights:
            self.q = weights[self.layer_id]['q']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_and_dequantize(pool, -128, 127)
        return qpool
    
    def collect(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return qpool, {self.layer_id: {'scale': spool}}

    def predict(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = quantize_predict(pool, self.q, -128, 127) # this only works because we use np.ceil(scales)
        return qpool
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict








        
        
        
