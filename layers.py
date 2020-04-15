
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

# if we want to train:
# https://stackoverflow.com/questions/55764694/how-to-use-gradient-override-map-in-tensorflow-2-0

def quantize(x, low, high):
    scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
    x = x / scale
    x = tf.floor(x)
    x = tf.clip_by_value(x, low, high)
    return x, scale

#############

class model:
    def __init__(self, layers):
        self.layers = layers
        self.ymax = 0.
        
    def train(self, x, scale):
        y = x
        for layer in self.layers:
            y = layer.train(y, scale, self.ymax, -1)
        return y
    
    def upto(self, x, n):
        # print (n)        
        y = x
        for layer in self.layers:
            y = layer.train(y, True, self.ymax, n)
        return y

    def predict(self, x):
        assert (False)

    def set_ymax(self):
        for layer in self.layers:
            self.ymax = tf.maximum(self.ymax, layer.ymax())
            
        self.ymax = 14.

    def get_weights(self):
        weights_dict = {}
        for layer in self.layers:
            weights_dict.update(layer.get_weights())
        return weights_dict
        
#############

class layer:
    layer_id = 0
    layer_num = 0
    
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
        self.layer_num = layer.layer_num
        layer.layer_num += 1
        
        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        
        self.noise = noise
        
        self.relu = relu
        
        self.max1 = 0.
        self.max2 = 0.
        
        self.ymax1 = 0.
        self.ymax2 = 0.
        self.ymax3 = 0.
        
        if weights:
            # print (self.layer_id, shape, weights[self.layer_id].keys())
            f, b, s, scale, z = weights[self.layer_id]['f'], weights[self.layer_id]['b'], weights[self.layer_id]['s'], weights[self.layer_id]['scale'], weights[self.layer_id]['z']
            assert (np.shape(f) == shape)
            self.f = tf.Variable(f, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.s = s
            self.z = z
            self.scale = tf.Variable(scale, dtype=tf.float32, trainable=False)
        else:
            assert (False)

    def train(self, x, scale, ymax, nlayer):
        if not scale:
            self.max1 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max1)
            self.max2 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max2)
            x_scale = 1
        elif self.layer_num > nlayer:
            x_scale = self.max2 / self.max1
        else:
            self.max2 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max2)
            x_scale = self.max2 / self.max1
    
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        
        if not scale:
            conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + self.b / self.scale * x_scale 
        else:
            conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + tf.round(self.b / self.scale * x_scale)
            self.b_scale = tf.round(self.b / self.scale * x_scale)

        if self.relu:
            out = tf.nn.relu(conv)
        else:
            out = conv

        if not scale:
            out = out * self.scale
            self.ymax1 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax1)
            self.ymax2 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax2)
            y_scale = 1
        elif self.layer_num > nlayer:
            y_scale = (127. / self.ymax2) * tf.minimum(1., (self.ymax() / ymax))
            self.y_scale = tf.round(1. / (y_scale * self.scale))
            out = out / tf.round(1. / (y_scale * self.scale))
            out = tf.clip_by_value(tf.round(out), -128, 127)
            self.ymax3 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax3)
        else:
            out = out * self.scale
            self.ymax2 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax2)
            y_scale = (127. / self.ymax2) * tf.minimum(1., (self.ymax() / ymax))
            out = out * y_scale
            # out = tf.round(out)
            self.ymax3 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax3)

        # if self.layer_num == nlayer: print (self.layer_num, self.k, self.ymax1, self.ymax2, self.ymax3)
        # if self.layer_num == nlayer: print (self.layer_num, x_scale, y_scale, tf.reduce_max(y_scale * self.scale), tf.reduce_max(self.b / self.scale * x_scale))

        return out
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)
        
    def ymax(self):
        return self.ymax1

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_num] = {'f': self.f.numpy(), 'b': self.b_scale.numpy(), 'y': self.y_scale.numpy()}
        return weights_dict
        
#############

class res_block1(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1
        self.layer_num = layer.layer_num
        layer.layer_num += 1

        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise

        self.ymax1 = 0.
        self.ymax2 = 0.

        if weights:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
        else:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=None)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=None, relu=False)

    def train(self, x, scale, ymax, nlayer):
        y1 = self.conv1.train(x, scale, ymax, nlayer)
        y2 = self.conv2.train(y1, scale, ymax, nlayer)        
        y3 = tf.nn.relu(y2 + x)
        return y3 

    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)
        
    def ymax(self):
        return tf.maximum(self.conv1.ymax(), self.conv2.ymax())

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict[self.layer_num] = {}
        return weights_dict

#############

class res_block2(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        # self.layer_id = layer.layer_id
        # layer.layer_id += 1
        self.layer_num = layer.layer_num
        layer.layer_num += 1

        self.f1 = f1
        self.f2 = f2
        self.p = p
        self.noise = noise

        self.ymax1 = 0.
        self.ymax2 = 0.
        
        if weights:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)
            self.conv3 = conv_block((1, 1, f1, f2), p, noise=None, weights=weights, relu=False)
        else:
            self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=None)
            self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=None, relu=False)
            self.conv3 = conv_block((1, 1, f1, f2), p, noise=None, weights=None, relu=False)

    def train(self, x, scale, ymax, nlayer):
        y1 = self.conv1.train(x, scale, ymax, nlayer)
        y2 = self.conv2.train(y1, scale, ymax, nlayer)
        y3 = self.conv3.train(x, scale, ymax, nlayer)
        y4 = tf.nn.relu(y2 + y3)
        return y4 

    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)
        
    def ymax(self):
        return tf.maximum(tf.maximum(self.conv1.ymax(), self.conv2.ymax()), self.conv3.ymax())

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        weights3 = self.conv3.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)
        weights_dict[self.layer_num] = {}
        return weights_dict

#############

class dense_block(layer):
    def __init__(self, isize, osize, noise, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        self.layer_num = layer.layer_num
        layer.layer_num += 1
    
        self.isize = isize
        self.osize = osize
        self.noise = noise
        
        self.max1 = 0.
        self.max2 = 0.
        
        if weights:
            w, b, s, scale, z = weights[self.layer_id]['w'], weights[self.layer_id]['b'], weights[self.layer_id]['s'], weights[self.layer_id]['scale'], weights[self.layer_id]['z']
            self.w = tf.Variable(w, dtype=tf.float32, trainable=False)
            self.b = tf.Variable(b, dtype=tf.float32, trainable=False)
            self.s = s
            self.z = z
            self.scale = tf.Variable(scale, dtype=tf.float32, trainable=False)
        else:
            self.w = tf.Variable(init_matrix(size=(self.isize, self.osize), init='glorot_uniform'), dtype=tf.float32)
            self.b = tf.Variable(np.zeros(shape=(self.osize)), dtype=tf.float32, trainable=False)
        
    def train(self, x, scale, ymax, nlayer):
        if not scale:
            self.max1 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max1)
            self.max2 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max2)
            x_scale = 1
        elif self.layer_num > nlayer:
            x_scale = self.max2 / self.max1
        else:
            self.max2 = tf.maximum(tf.reduce_max(tf.abs(x)), self.max2)
            x_scale = self.max2 / self.max1

        x = tf.reshape(x, (-1, self.isize))
        # fc = tf.matmul(x, self.w) + self.b / self.scale * x_scale
        # fc = fc * self.scale
        
        if not scale:
            fc = tf.matmul(x, self.w) + self.b / self.scale * x_scale
            fc = fc * self.scale
        else:
            fc = tf.matmul(x, self.w) + tf.round(self.b / self.scale * x_scale)
            fc = fc / tf.round(1. / self.scale)
            self.b_scale = tf.round(self.b / self.scale * x_scale)
            self.y_scale = tf.round(1. / self.scale)
        
        # if self.layer_num == nlayer: print (1. / self.scale)
        return fc
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)
        
    def ymax(self):
        return 0.

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_num] = {'w': self.w.numpy(), 'b': self.b_scale.numpy(), 'y': self.y_scale.numpy()}
        return weights_dict

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_num = layer.layer_num
        layer.layer_num += 1

        self.s = s
        self.p = p
        
    def train(self, x, scale, ymax, nlayer):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)
        
    def ymax(self):
        return 0.

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_num] = {}
        return weights_dict

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_num = layer.layer_num
        layer.layer_num += 1

        self.s = s
        self.p = p
        
        self.ymax1 = 0.
        self.ymax2 = 0.
        
    def train(self, x, scale, ymax, nlayer):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
    
    def collect(self, x):
        assert (False)

    def predict(self, x):
        assert (False)

    def ymax(self):
        return self.ymax1

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_num] = {}
        return weights_dict








        
        
        
