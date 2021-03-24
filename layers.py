
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
        return y
        
    def get_weights(self):
        weights_dict = {}
        s = 1
        for layer in self.layers:
            w, s = layer.get_weights(s)
            weights_dict.update(w)
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
    def __init__(self, shape, p, weights=None, relu=True, train=True, quantize=True):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.pad = self.k // 2
        self.p = p

        self.relu_flag = relu
        self.train_flag = train
        self.quantize_flag = quantize

        if self.train_flag:
            self.total = 0
            self.std = np.zeros(shape=self.f2)
            self.mean = np.zeros(shape=self.f2)
            self.q_sum = 0.
            if weights:
                f = weights[self.weight_id]['f']
                b = weights[self.weight_id]['b']
                g = weights[self.weight_id]['g']
                self.f = tf.Variable(f, dtype=tf.float32)
                self.b = tf.Variable(b, dtype=tf.float32)
                self.g = tf.Variable(g, dtype=tf.float32)
            else:
                self.f = tf.Variable(init_filters(size=[self.k,self.k,self.f1,self.f2], init='glorot_uniform'), dtype=tf.float32, name='f_%d' % (self.layer_id))
                self.b = tf.Variable(np.zeros(shape=(self.f2)), dtype=tf.float32, name='b_%d' % (self.layer_id))
                self.g = tf.Variable(np.ones(shape=(self.f2)), dtype=tf.float32, name='g_%d' % (self.layer_id))
        else:
            f = weights[self.layer_id]['f']
            b = weights[self.layer_id]['b']
            q = weights[self.layer_id]['q']
            sf = weights[self.layer_id]['sf']
            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = q
            self.sf = sf

    def train(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)

        qf, sf = quantize_and_dequantize(fold_f, -128, 127)
        qb = fold_b

        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + qb
        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        if self.quantize_flag:
            if self.relu_flag: qout, _ = quantize_and_dequantize(out, 0, 255)
            else:              qout, _ = quantize_and_dequantize(out, -128, 127)
        else:                  qout = out

        return qout
    
    def collect(self, x, s):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID')
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        std = tf.sqrt(var + 1e-3)
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)

        qf, sf = quantize(fold_f, -128, 127)
        qb = fold_b / sf
        
        conv = tf.nn.conv2d(x_pad, qf, [1,self.p,self.p,1], 'VALID') + qb
        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        self.std += std.numpy()
        self.mean += mean.numpy()
        self.total += 1

        if self.quantize_flag:
            if self.relu_flag: qout, sout = quantize(out, 0, 255)
            else:              qout, sout = quantize(out, -128, 127)
            self.q_sum += sout.numpy()
            scale = sout * sf
        else:
            qout = out
            scale = sf

        return qout, scale

    def predict(self, x, s):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + self.b

        if self.relu_flag: out = tf.nn.relu(conv)
        else:              out = conv

        if self.quantize_flag:
            if self.relu_flag: qout = quantize_scale(out, self.q, 0, 255)
            else:              qout = quantize_scale(out, self.q, -128, 127)
            scale = self.sf * self.q
        else:
            qout = out
            scale = self.sf

        return qout, scale
        
    def get_weights(self, s):
        weights_dict = {}

        std = self.std / self.total
        mean = self.mean / self.total
        q = self.q_sum / self.total

        fold_f = (self.g * self.f) / std
        qf, sf = quantize(fold_f, -128, 127)

        fold_b = self.b - ((self.g * mean) / std)
        qb = fold_b / sf

        weights_dict[self.layer_id] = {'f': qf.numpy(), 'b': qb.numpy(), 'q': q, 'sf': sf.numpy()}
        scale = sf * q if self.quantize_flag else sf
        return weights_dict, scale.numpy()

    def get_params(self):
        return [self.f, self.b, self.g]

#############

class dense_block(layer):
    def __init__(self, isize, osize, weights=None, train=True):
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.weight_id = layer.weight_id
        layer.weight_id += 1

        self.isize = isize
        self.osize = osize

        self.train_flag = train

        if self.train_flag:
            self.total = 0
            self.q_sum = 0.
            if weights:
                w = weights[self.weight_id]['w']
                b = weights[self.weight_id]['b']
                self.w = tf.Variable(w, dtype=tf.float32)
                self.b = tf.Variable(b, dtype=tf.float32)
            else:
                self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32, name='w_%d' % (self.layer_id))
                self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, name='b_%d' % (self.layer_id))
        else:
            w = weights[self.layer_id]['w']
            b = weights[self.layer_id]['b']
            q = weights[self.layer_id]['q']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.q = q

    def train(self, x):
        qw, sw = quantize_and_dequantize(self.w, -128, 127)
        qb = self.b

        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + qb
        qfc, _ = quantize_and_dequantize(fc, -128, 127)
        return qfc
    
    def collect(self, x, s):
        qw, sw = quantize(self.w, -128, 127)
        qb = self.b / sw
        
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, qw) + qb
        qfc, sfc = quantize(fc, -128, 127)

        self.q_sum += sfc.numpy()
        self.total += 1

        return qfc, sfc

    def predict(self, x, s):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        qfc = quantize_scale(fc, self.q, -128, 127)
        return qfc, self.q
        
    def get_weights(self, s):
        weights_dict = {}
        qw, sw = quantize(self.w, -128, 127)
        qb = self.b / sw
        q = self.q_sum / self.total
        weights_dict[self.layer_id] = {'w': qw.numpy(), 'b': qb.numpy(), 'q': q, 'sw': sw.numpy()}
        return weights_dict, sw * q
        
    def get_params(self):
        return [self.w, self.b]

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x, s):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, 0, 255)
        return pool, s

    def predict(self, x, s):
        return self.collect(x, s)
        
    def get_weights(self, s):
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict, s
        
    def get_params(self):
        return []

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x, s):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, 0, 255)
        return pool, s

    def predict(self, x, s):
        return self.collect(x, s)
        
    def get_weights(self, s):
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict, s

    def get_params(self):
        return []

#############

class res_block1(layer):
    def __init__(self, f1, f2, p, weights=None, train=True):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.train_flag = train

        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights, relu=True,  train=train)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False, train=train)

        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.total = 0
        self.q_sum = 0.

        if not train:
            self.q = weights[self.layer_id]['q']
            self.s = weights[self.layer_id]['s']
            self.s2 = weights[self.layer_id]['s2']

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(x + y2)
        qout, _ = quantize_and_dequantize(y3, 0, 255)
        return qout

    def collect(self, x, s):
        y1, s1 = self.conv1.collect(x, s)
        y2, s2 = self.conv2.collect(y1, s1)
        y3 = tf.nn.relu(s*x + s2*y2)
        out, sout = quantize(y3, 0, 255)

        self.q_sum += sout.numpy()
        self.total += 1

        return out, sout

    def predict(self, x, s):
        y1, s1 = self.conv1.predict(x, s)
        y2, s2 = self.conv2.predict(y1, s1)
        #print (self.s, s)
        #print (self.s2, s2)
        y3 = tf.nn.relu(self.s*x + self.s2*y2)
        out = quantize_scale(y3, self.q, 0, 255)
        return out, self.q

    def get_weights(self, s):
        weights_dict = {}
        weights1, s1 = self.conv1.get_weights(s)
        weights2, s2 = self.conv2.get_weights(s1)
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)

        q = self.q_sum / self.total
        weights_dict[self.layer_id] = {'q': q, 's2': s2, 's': s}
        return weights_dict, q
        
    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        return params

#############

class res_block2(layer):
    def __init__(self, f1, f2, p, weights=None, train=True):

        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.train_flag = train

        self.conv1 = conv_block((3, 3, f1, f2), p, weights=weights, relu=True,  train=train)
        self.conv2 = conv_block((3, 3, f2, f2), 1, weights=weights, relu=False, train=train, quantize=False)
        self.conv3 = conv_block((1, 1, f1, f2), p, weights=weights, relu=False, train=train, quantize=False)
        
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.total = 0
        self.q_sum = 0.

        if not train:
            self.q = weights[self.layer_id]['q']
            self.s2 = weights[self.layer_id]['s2']
            self.s3 = weights[self.layer_id]['s3']

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = self.conv3.train(x)
        y4 = tf.nn.relu(y2 + y3)
        out, _ = quantize_and_dequantize(y4, 0, 255)
        return out

    def collect(self, x, s):
        y1, s1 = self.conv1.collect(x, s)
        y2, s2 = self.conv2.collect(y1, s1)
        y3, s3 = self.conv3.collect(x, s)
        y4 = tf.nn.relu(s2*y2 + s3*y3)
        out, sout = quantize(y4, 0, 255)

        self.q_sum += sout.numpy()
        self.total += 1

        return out, sout

    def predict(self, x, s):
        y1, s1 = self.conv1.predict(x, s)
        y2, s2 = self.conv2.predict(y1, s1)
        y3, s3 = self.conv3.predict(x, s)
        #print (self.s2, s2)
        #print (self.s3, s3)
        y4 = tf.nn.relu(self.s2*y2 + self.s3*y3)
        out = quantize_scale(y4, self.q, 0, 255)
        return out, self.q

    def get_weights(self, s):
        weights_dict = {}
        weights1, s1 = self.conv1.get_weights(s)
        weights2, s2 = self.conv2.get_weights(s1)
        weights3, s3 = self.conv3.get_weights(s2)
       
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)

        q  = self.q_sum / self.total
        weights_dict[self.layer_id] = {'q': q, 's2': s2, 's3': s3}
        return weights_dict, q

    def get_params(self):
        params = []
        params.extend(self.conv1.get_params())
        params.extend(self.conv2.get_params())
        params.extend(self.conv3.get_params())
        return params

#############



        
        
        
