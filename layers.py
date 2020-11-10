
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
        for layer in self.layers:
            y = layer.collect(y)
        return y

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
    def __init__(self, f1, f2, weights=None, train=True):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.f1 = f1
        self.f2 = f2

        self.train_flag = train

        if self.train_flag:
            self.total = 0
            self.std = np.zeros(shape=self.f2)
            self.mean = np.zeros(shape=self.f2)
            self.scale = 0.
            if weights:
                f = weights[self.layer_id]['f']
                b = weights[self.layer_id]['b']
                g = weights[self.layer_id]['g']
                self.f = tf.Variable(f, dtype=tf.float32)
                self.b = tf.Variable(b, dtype=tf.float32)
                self.g = tf.Variable(g, dtype=tf.float32)
            else:
                self.f = tf.Variable(init_filters(size=[3,3,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32)
                self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32)
                self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32)
        else:
            f = weights[self.layer_id]['f']
            b = weights[self.layer_id]['b']
            q = weights[self.layer_id]['q']
            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = q

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

        qout = quantize_and_dequantize(relu, -128, 127)
        return qout
    
    def collect(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME') # there is no bias when we have bn.
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)

        qf, sf = quantize(fold_f, -128, 127)
        qb = quantize_predict(fold_b, sf, -2**24, 2**24-1)
        
        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + qb
        relu = tf.nn.relu(conv)

        qout, sout = quantize(relu, -128, 127)

        n, h, w, c = np.shape(x)
        self.std += std.numpy()
        self.mean += mean.numpy()
        self.scale += sout.numpy()
        self.total += 1

        return qout, sf * sout

    def predict(self, x):
        conv = tf.nn.conv2d(x, self.f, [1,1,1,1], 'SAME') + self.b
        relu = tf.nn.relu(conv)
        qout = quantize_predict(relu, self.q, -128, 127)
        return qout
        
    def get_weights(self):
        weights_dict = {}

        std = self.std / self.total
        mean = self.mean / self.total
        q = self.scale / self.total

        fold_f = (self.g * self.f) / std
        qf, sf = quantize(fold_f, -128, 127)

        fold_b = self.b - ((self.g * mean) / std)
        qb = quantize_predict(fold_b, sf, -2**24, 2**24-1) # can probably leave b as a float.

        weights_dict[self.layer_id] = {'f': qf.numpy(), 'b': qb.numpy(), 'q': q}
        return weights_dict

    def get_params(self):
        return [self.f, self.b, self.g]

#############

class dense_block(layer):
    def __init__(self, isize, osize, weights=None, train=True):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.isize = isize
        self.osize = osize

        self.train_flag = train

        if self.train_flag:
            self.total = 0
            self.scale = 0.
            if weights:
                w = weights[self.layer_id]['w']
                b = weights[self.layer_id]['b']
                self.w = tf.Variable(w, dtype=tf.float32)
                self.b = tf.Variable(b, dtype=tf.float32)
            else:
                self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
                self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
        else:
            w = weights[self.layer_id]['w']
            b = weights[self.layer_id]['b']
            q = weights[self.layer_id]['q']
            self.w = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = q

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

        self.scale += sfc.numpy()
        self.total += 1

        return qfc, sfc

    def predict(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        qfc = quantize_predict(fc, self.q, -128, 127)
        return qfc
        
    def get_weights(self):
        weights_dict = {}        
        qw, sw = quantize(self.w, -128, 127)
        qb = quantize_predict(self.b, sw, -2**24, 2**24-1) # can probably leave b as a float.
        q = self.scale / self.total
        weights_dict[self.layer_id] = {'w': qw.numpy(), 'b': qb.numpy(), 'q': q}
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
        return pool, 1

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
        return pool, 1

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


#############

class res_block1(layer):
    def __init__(self, f1, f2, p, weights=None):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False)

        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.total = 0
        self.scale = 0.

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(x + y2)
        qout = quantize_and_dequantize(y3, -128, 127)
        return qout

    def collect(self, x):
        y1, s1 = self.conv1.collect(x)
        y2, s2 = self.conv2.collect(y1)
        y3 = tf.nn.relu(x + s2 * y2)
        out, sout = quantize(y3, -128, 127)

        self.scale += sout.numpy()
        self.total += 1

        return out, sout

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        return weights_dict
        
    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        return params

#############

class res_block2(layer):
    def __init__(self, f1, f2, p, weights=None):

        self.f1 = f1
        self.f2 = f2
        self.p = p
        
        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, weights=weights, relu=False)
        
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.total = 0
        self.scale = 0.

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = self.conv3.train(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4

    def collect(self, x):
        y1, s1 = self.conv1.collect(x)
        y2, s2 = self.conv2.collect(y1)
        y3, s3 = self.conv3.collect(x)
        y4 = tf.nn.relu(s1 * s2 * y2 + s3 * y3)
        out, sout = quantize(y4, -128, 127)

        self.scale += sout.numpy()
        self.total += 1

        return out, sout

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        weights3 = self.conv3.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)
        return weights_dict

    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        params.extend(self.conv3.get_params())
        return params

#############



        
        
        
