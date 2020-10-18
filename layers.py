
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

# https://stackoverflow.com/questions/59656219/override-tf-floor-gradient

# this would also work:
# https://www.tensorflow.org/api_docs/python/tf/grad_pass_through

@tf.custom_gradient
def floor_no_grad(x):

    def grad(dy):
        return dy
    
    return tf.floor(x), grad
    

def quantize_and_dequantize(x, low, high):
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Floor": "Identity"}):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = floor_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    x = x * scale
    return x

def quantize(x, low, high):
    # g = tf.get_default_graph()
    # with g.gradient_override_map({"Floor": "Identity"}):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = floor_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    return x, scale
    
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
        y = x
        stats = {}
        for layer in self.layers:
            y, stat = layer.collect(y)
            stats.update(stat)
        return y, stats

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
        
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params

    def save(self, name):
        weights = self.get_weights()
        np.save(name, weights)

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
    def __init__(self, f1, f2, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        if weights:
            print (weights[self.layer_id].keys())
            f     = weights[self.layer_id]['f']
            g     = weights[self.layer_id]['g']
            b     = weights[self.layer_id]['b']
            scale = weights[self.layer_id]['scale']
            std   = weights[self.layer_id]['std']
            mean  = weights[self.layer_id]['mean']

            f = f * g / std
            b = b - (g / std) * mean

            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = scale
            
        else:
            self.f = tf.Variable(init_filters(size=[3,3,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32)
            self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32)
            self.q = None

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME') # there is no bias when we have bn.
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        qf = quantize_and_dequantize(fold_f, -128, 127)
        # qb = quantize_and_dequantize(fold_b, -128, 127) 
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + fold_b
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')

        qpool = quantize_and_dequantize(pool, -128, 127)
        return qpool
    
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

        return qpool, {self.layer_id: {'std': std, 'mean': mean, 'scale': spool}}

    def predict(self, x):
        qf, sf = quantize(self.f, -128, 127)
        qb = quantize_predict(self.b, sf, -2**24, 2**24-1)

        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        
        qpool = quantize_predict(pool, self.q, -128, 127)
        return qpool
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict

    def get_params(self):
        return [self.f, self.b, self.g]

#############

class dense_block(layer):
    def __init__(self, isize, osize, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.isize = isize
        self.osize = osize

        if weights:
            print (weights[self.layer_id].keys())
            w, b, q = weights[self.layer_id]['w'], weights[self.layer_id]['b'], weights[self.layer_id]['scale']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = q
        else:
            self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
            self.q = None
        
    def train(self, x):
        qw = quantize_and_dequantize(self.w, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + self.b
        qfc = quantize_and_dequantize(fc, -128, 127)
        return qfc
    
    def collect(self, x):
        qw, sw = quantize(self.w, -128, 127)
        qb = quantize_predict(self.b, sw, -2**24, 2**24-1)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + qb
        qfc, sfc = quantize(fc, -128, 127)
        # return qfc, [sfc]
        return qfc, {self.layer_id: {'scale': sfc}}

    def predict(self, x):
        qw = self.w
        qb = self.b
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + qb
        qfc = quantize_predict(fc, self.q, -128, 127)
        return qfc
        
    def get_weights(self):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
    
        weights_dict = {}
        weights_dict[self.layer_id] = {'w': qw, 'b': qb}
        
        return weights_dict
        
    def get_params(self):
        return [self.w]

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
        if weights:
            print ('pool', weights.keys())
            self.q = weights['y']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_and_dequantize(pool, -128, 127)
        return qpool
    
    def collect(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        # return pool, [spool]
        return pool, {}

    def predict(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_predict(pool, scale, -128, 127) # this only works because we use np.ceil(scales)
        return qpool
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict
        
    def get_params(self):
        return []

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
        if weights:
            print ('pool', weights.keys())
            self.q = weights['y']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool 
        return qpool
    
    def collect(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        # return pool, [spool]
        return pool, {}

    def predict(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool 
        return qpool
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict

    def get_params(self):
        return []






        
        
        
