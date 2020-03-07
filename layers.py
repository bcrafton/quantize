
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

def quantize(x, low, high):
    scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
    x = x / scale
    x = tf.floor(x)
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
        scale = []
        y = x 
        for layer in self.layers:
            y, s = layer.collect(y)
            scale.append(s)
        return scale

    def predict(self, x, scale):
        y = x
        for l in range(len(self.layers)):
            y = self.layers[l].predict(y, scale[l])
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

    def predict(self, x, scale):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, f1, f2, p):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.f = tf.Variable(init_filters(size=[3,3,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32)
        self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32, trainable=False)
        
    def train(self, x):
        qf = tf.quantization.quantize_and_dequantize(input=self.f, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        qb = tf.quantization.quantize_and_dequantize(input=self.b, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') # + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        qpool = tf.quantization.quantize_and_dequantize(input=pool, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        return qpool
    
    def collect(self, x):
        qf, _ = quantize(self.f, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') # + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        qpool, spool = quantize(pool, -128, 127)
        return qpool, spool

    def predict(self, x, scale):
        qf, _ = quantize(self.f, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') # + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        qpool = quantize_predict(pool, scale, -128, 127)
        return qpool
        
    def get_weights(self):
        qf, _ = quantize(self.f, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
    
        weights_dict = {}
        weights_dict[self.layer_id] = (qf, qb)
        
        return weights_dict
        
#############

class dense_block(layer):
    def __init__(self, isize, osize):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.isize = isize
        self.osize = osize
        self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
        self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
        
    def train(self, x):
        qw = tf.quantization.quantize_and_dequantize(input=self.w, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        qb = tf.quantization.quantize_and_dequantize(input=self.b, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) # + qb
        qfc = tf.quantization.quantize_and_dequantize(input=fc, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        return qfc
    
    def collect(self, x):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) # + qb
        qfc, sfc = quantize(fc, -128, 127)
        return qfc, sfc

    def predict(self, x, scale):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) # + qb
        qfc = quantize_predict(fc, scale, -128, 127)
        return qfc
        
    def get_weights(self):
        qw, _ = quantize(self.w, -128, 127)
        qb, _ = quantize(self.b, -128, 127)
    
        weights_dict = {}
        weights_dict[self.layer_id] = (qw, qb)
        
        return weights_dict

#############

class avg_pool(layer):
    def __init__(self, s, p):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = tf.quantization.quantize_and_dequantize(input=pool, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        return qpool
    
    def collect(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return qpool, spool

    def predict(self, x, scale):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = quantize_predict(pool, scale, -128, 127) # this only works because we use np.ceil(scales)
        return qpool
        
    def get_weights(self):    
        weights_dict = {}
        return weights_dict

#############










        
        
        
