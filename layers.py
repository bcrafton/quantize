
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

class model:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        
    def train(self, x):
        y = x
        for layer in self.layers:
            y = layer.train(y)
        return y
    
    def collect(self, x):
        y = x
        for layer in self.layers:
            y = layer.collect(y)
        return y

    def predict(self, x, scale):
        y = x
        for layer in self.layers:
            y = layer.predict(y, scale)
        return y
        
#############

class layer:
    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)
    
    def collect(self, x):
        assert(False)

    def predict(self, x, scale):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, f1, f2, p):
        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.f = tf.Variable(init_filters(size=[3,3,self.f1,self.f2], init='alexnet'), dtype=tf.float32)
        
    def train(self, x):
        qf = tf.quantization.quantize_and_dequantize(input=self.f, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME')
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        qpool = tf.quantization.quantize_and_dequantize(input=pool, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        return qpool
    
    def collect(self, x):
        return self.train(x)

    def predict(self, x, scale):
        return self.train(x)
        
#############

class dense_block(layer):
    def __init__(self, isize, osize):
        self.isize = isize
        self.osize = osize
        self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='alexnet'), dtype=tf.float32)
        
    def train(self, x):
        x = tf.reshape(x, (-1, self.isize))
        qw = tf.quantization.quantize_and_dequantize(input=self.w, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        fc = tf.matmul(x, qw)
        qfc = tf.quantization.quantize_and_dequantize(input=fc, input_min=0, input_max=0, signed_input=True, num_bits=8, range_given=False)
        return qfc
    
    def collect(self, x):
        return self.train(x)

    def predict(self, x, scale):
        return self.train(x)

#############

class avg_pool(layer):
    def __init__(self, s, p):
        self.s = s
        self.p = p
        
    def train(self, x):        
        return tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
    
    def collect(self, x):
        return tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")

    def predict(self, x, scale):
        return tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")

#############










        
        
        
