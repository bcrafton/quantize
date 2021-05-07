
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

@tf.custom_gradient
def round_no_grad(x):
    def grad(dy):
        return dy
    return tf.round(x), grad

def quantize_and_dequantize(x, low, high):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = round_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    x = x * scale
    return x, scale

def quantize_and_dequantize_scale(x, scale, low, high):
    x = x / scale
    x = round_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    x = x * scale
    return x

def quantize(x, low, high):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = round_no_grad(x)
    x = tf.clip_by_value(x, low, high)
    return x, scale
    
def quantize_scale(x, scale, low, high):
    x = x / scale
    x = round_no_grad(x)
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
        s = tf.constant(1.)
        for layer in self.layers:
            y, s = layer.collect(y, s)
        return y

    def predict(self, x):
        y = x
        s = tf.constant(1.)
        for layer in self.layers:
            y, s = layer.predict(y, s)
            assert (np.max(y) <   128)
            assert (np.min(y) >= -128)
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
    weight_id = 0

    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)
    
    def collect(self, x, s):
        assert(False)

    def predict(self, x, s):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, weights=None, relu=True, train=True, quantize=True, bits=8):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.p = p

        self.relu_flag = relu
        self.train_flag = train
        self.quantize_flag = quantize

        self.bits = bits
        self.HIGH = 2 ** (self.bits - 1) - 1
        self.LOW  = -2 ** (self.bits - 1)

        if self.train_flag:
            self.total = 0
            self.std = np.zeros(shape=self.f2)
            self.mean = np.zeros(shape=self.f2)
            self.q_sum = 0.
            if weights:
                f = weights[self.weight_id]['f']
                g = weights[self.weight_id]['g']
                self.f = tf.Variable(f, dtype=tf.float32)
                self.g = tf.Variable(g, dtype=tf.float32)
            else:
                f = init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform')
                g = np.ones(shape=(self.f2))
                self.f = tf.Variable(f, dtype=tf.float32, name='f_%d' % (self.layer_id))
                self.g = tf.Variable(g, dtype=tf.float32, name='g_%d' % (self.layer_id))
        else:
            f = weights[self.layer_id]['f']
            q = weights[self.layer_id]['q']
            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.q = q

    def train(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,self.p,self.p,1], 'VALID')
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        qf, sf = quantize_and_dequantize(fold_f, self.LOW, self.HIGH)

        conv = tf.nn.conv2d(x, qf, [1,self.p,self.p,1], 'VALID')
        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        if self.quantize_flag: qout, _ = quantize_and_dequantize(out, -128, 127)
        else:                  qout = out

        return qout
    
    def collect(self, x, s):
        conv = tf.nn.conv2d(x, self.f, [1,self.p,self.p,1], 'VALID')
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        qf, sf = quantize(fold_f, self.LOW, self.HIGH)
        
        conv = tf.nn.conv2d(x, qf, [1,self.p,self.p,1], 'VALID')
        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        self.std += std.numpy()
        self.mean += mean.numpy()
        self.total += 1

        if self.quantize_flag:
            qout, sout = quantize(out, -128, 127)
            self.q_sum += sout.numpy()
            scale = sout * sf
        else:
            qout = out
            scale = sf

        return qout, scale

    def predict(self, x, s):
        conv = tf.nn.conv2d(x, self.f, [1,self.p,self.p,1], 'VALID')

        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        if self.quantize_flag:
            qout = quantize_scale(out, self.q, -128, 127)
            scale = self.q
        else:
            qout = out
            scale = tf.constant(1.)

        return qout, scale
        
    def get_weights(self):
        weights_dict = {}

        std = self.std / self.total
        mean = self.mean / self.total
        q = self.q_sum / self.total

        fold_f = (self.g * self.f) / std
        qf, sf = quantize(fold_f, self.LOW, self.HIGH)

        print (self.layer_id, q)
        weights_dict[self.layer_id] = {'f': qf.numpy().astype(int), 'q': int(q)}
        return weights_dict

    def get_params(self):
        return [self.f, self.g]

#############

class dense_block(layer):
    def __init__(self, isize, osize, weights=None, train=True, bits=8):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.isize = isize
        self.osize = osize

        self.train_flag = train
        self.bits = bits
        self.HIGH = 2 ** (self.bits - 1) - 1
        self.LOW  = -2 ** (self.bits - 1)

        if self.train_flag:
            self.total = 0
            self.q_sum = 0.
            if weights:
                w = weights[self.weight_id]['w']
                self.w = tf.Variable(w, dtype=tf.float32)
            else:
                self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32, name='w_%d' % (self.layer_id))
        else:
            w = weights[self.layer_id]['w']
            q = weights[self.layer_id]['q']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.q = q

    def train(self, x):
        qw, sw = quantize_and_dequantize(self.w, self.LOW, self.HIGH)

        x = tf.transpose(x, (0,3,1,2))
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw)
        qfc, _ = quantize_and_dequantize(fc, -128, 127)
        return qfc
    
    def collect(self, x, s):
        qw, sw = quantize(self.w, self.LOW, self.HIGH)
        
        x = tf.transpose(x, (0,3,1,2))
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw)
        qfc, sfc = quantize(fc, -128, 127)

        self.q_sum += sfc.numpy()
        self.total += 1

        return qfc, sfc

    def predict(self, x, s):
        x = tf.transpose(x, (0,3,1,2))
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w)
        qfc = quantize_scale(fc, self.q, -128, 127)
        return qfc, self.q
        
    def get_weights(self):
        weights_dict = {}
        qw, sw = quantize(self.w, self.LOW, self.HIGH)
        q = self.q_sum / self.total
        print (self.layer_id, q)
        weights_dict[self.layer_id] = {'w': qw.numpy().astype(int), 'q': int(q)}
        return weights_dict
        
    def get_params(self):
        return [self.w]

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        assert (False)
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x, s):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return pool, s

    def predict(self, x, s):
        return self.collect(x, s)
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict
        
    def get_params(self):
        return []

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        assert (False)
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x, s):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return pool, s

    def predict(self, x, s):
        return self.collect(x, s)
        
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict

    def get_params(self):
        return []

#############



        
        
        
