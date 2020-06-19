
import numpy as np
np.set_printoptions(threshold=1000)

import tensorflow as tf

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

#############

def quantize(x, low, high):
    scale = tf.reduce_max(tf.abs(x)) / high
    x = x / scale
    x = tf.round(x)
    x = tf.clip_by_value(x, low, high)
    return x, scale

#############

def quantize_and_dequantize_np(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    x = x * scale
    return x, scale

def quantize_np(x, low, high):
    scale = np.max(np.absolute(x)) / high
    x = x / scale
    x = np.round(x)
    x = np.clip(x, low, high)
    return x, scale
    
#############

class model:
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x):
        assert (False)
    
    def collect(self, x):
        assert (False)

    def predict(self, x, q=False, l=0):
        y = x
        for layer in self.layers:
            y = layer.predict(y, q, l)
        return y
        
    def get_weights(self):
        weights_dict = {}
        for layer in self.layers:
            weights_dict.update(layer.get_weights())
        return weights_dict
        
#############

class layer:
    weight_id = 0
    layer_id = 0
    
    def __init__(self):
        assert(False)
        
    def train(self, x):        
        assert(False)
    
    def collect(self, x):
        assert(False)

    def predict(self, x, q=False, l=0):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, noise, weights=None, relu=True):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.k, _, self.f1, self.f2 = shape
        self.p = p
        self.pad = self.k // 2
        self.relu = relu
        
        if 'g' in weights[self.weight_id].keys():
            f, b, g, mean, var = weights[self.weight_id]['f'], weights[self.weight_id]['b'], weights[self.weight_id]['g'], weights[self.weight_id]['mean'], weights[self.weight_id]['var']
            
            var = np.sqrt(var + 1e-5)
            f = f * (g / var)
            b = b - (g / var) * mean

            #############################
            
            if self.weight_id == 0:
                std = np.array([0.229, 0.224, 0.225]) * 255. / 2.
                f = f / np.reshape(std, (3,1))

                mean = np.array([0.485, 0.456, 0.406]) * 255. / 2.
                expand_mean = np.ones(shape=(7,7,3)) * mean
                expand_mean = expand_mean.flatten()

                b = b - (expand_mean @ np.reshape(f, (7*7*3, 64)))
            
            #############################

            qf, scale = quantize_np(f, -128, 127)
            qb = b / scale
            
            self.f = tf.Variable(qf, dtype=tf.float32)
            self.b = tf.Variable(qb, dtype=tf.float32)
            self.scale = tf.Variable(scale, dtype=tf.float32)
        else:
            f, b = weights[self.weight_id]['f'], weights[self.weight_id]['b']

            qf, scale = quantize_np(f, -128, 127)
            qb = b / scale
            
            self.f     = tf.Variable(qf,    dtype=tf.float32)
            self.b     = tf.Variable(qb,    dtype=tf.float32)
            self.scale = tf.Variable(scale, dtype=tf.float32)
            
        assert (np.shape(f) == shape)
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

    def predict(self, x, q=False, l=0):
        if q:
            self.x2 = 127
            x_scale = self.x2 / self.x1
        else:
            self.x1 = tf.maximum(tf.reduce_max(tf.abs(x)), self.x1)
            x_scale = 1

        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + self.b * x_scale

        if self.relu:
            out = tf.nn.relu(conv)
        else:
            out = conv

        '''
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
        else:
            out = out * self.scale
            self.ymax2 = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax2)
            y_scale = (127. / self.ymax2) * tf.minimum(1., (self.ymax() / ymax))
            out = out * y_scale
        '''

        if q and (l > self.layer_id):
            self.y2 = tf.maximum(tf.reduce_max(tf.abs(out * self.scale)), self.y2)
            y_scale = self.scale * 127 / self.y2
            # y_scale = 1 / y_scale
            # y_scale = tf.round(y_scale)
            # y_scale = 1 / y_scale
            # print (self.layer_id, y_scale)
        elif q:
            out = out * self.scale
            y_scale = 1
        else:
            out = out * self.scale
            self.y1 = tf.maximum(tf.reduce_max(tf.abs(out)), self.y1)
            y_scale = 1
            
        out = out * y_scale
        return out

    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict

#############

class res_block1(layer):
    def __init__(self, f1, f2, p, noise, weights=None):
        
        self.f1 = f1
        self.f2 = f2
        self.p = p

        self.conv1 = conv_block((3, 3, f1, f2), p, noise=None, weights=weights)
        self.conv2 = conv_block((3, 3, f2, f2), 1, noise=None, weights=weights, relu=False)

        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.ymax = 0

    def predict(self, x, q=False, l=0):
        y1 = self.conv1.predict(x, q, l)
        y2 = self.conv2.predict(y1, q, l)
        
        if q and (l > self.layer_id):
            self.scale = self.conv2.y1 / self.conv1.x1
            out = tf.nn.relu(x + self.scale * y2)
            self.ymax = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax)
            out = out * (127 / self.ymax)
        else:
            out = tf.nn.relu(x + y2)

        return out

    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
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
        
        self.layer_id = layer.layer_id
        layer.layer_id += 1

        self.ymax = 0

    def predict(self, x, q=False, l=0):
        y1 = self.conv1.predict(x, q, l)
        y2 = self.conv2.predict(y1, q, l)
        y3 = self.conv3.predict(x, q, l)

        if q and (l > self.layer_id):
            self.scale = self.conv3.y1 / self.conv2.y1
            out = tf.nn.relu(y2 + self.scale * y3)
            self.ymax = tf.maximum(tf.reduce_max(tf.abs(out)), self.ymax)
            out = out * (127 / self.ymax)
        else:
            out = tf.nn.relu(y2 + y3)
            
        return out
        
    def get_weights(self):
        weights_dict = {}
        weights1 = self.conv1.get_weights()
        weights2 = self.conv2.get_weights()
        weights3 = self.conv3.get_weights()
        
        weights_dict.update(weights1)
        weights_dict.update(weights2)
        weights_dict.update(weights3)
        return weights_dict

#############

class dense_block(layer):
    def __init__(self, isize, osize, noise, weights=None):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.isize = isize
        self.osize = osize
        
        w, b = weights[self.weight_id]['w'], weights[self.weight_id]['b']
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def predict(self, x, q=False, l=0):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) # + self.b
        return fc
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'w': self.w, 'b': self.b}
        return weights_dict

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
    
        self.s = s
        self.p = p

    def predict(self, x, q=False, l=0):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
        
    def get_weights(self):    
        pass

#############

class max_pool(layer):
    def __init__(self, s, p, weights=None):
        self.s = s
        self.p = p
        
    def predict(self, x, q=False, l=0):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        return pool
        
    def get_weights(self):    
        pass






        
        
        
