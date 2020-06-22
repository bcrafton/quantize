
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
    
#############

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
        return y, scale

    def predict(self, x, scale):
        y = x
        for l in range(len(self.layers)):
            y = self.layers[l].predict(y, scale[l])
        return y
        
    def inference(self, x):
        y = x
        for layer in self.layers:
            y = layer.inference(y)
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

    def predict(self, x, scale):
        assert(False)
        
    def get_weights(self):
        assert(False)
        
#############
        
class conv_block(layer):
    def __init__(self, shape, p, weights=None, relu=True):
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
            
            '''
            if self.weight_id == 0:
                std = np.array([0.229, 0.224, 0.225]) * 255. / 2.
                f = f / np.reshape(std, (3,1))

                mean = np.array([0.485, 0.456, 0.406]) * 255. / 2.
                expand_mean = np.ones(shape=(7,7,3)) * mean
                expand_mean = expand_mean.flatten()

                b = b - (expand_mean @ np.reshape(f, (7*7*3, 64)))
            '''
            
            #############################

            # qf, scale = quantize_np(f, -128, 127)
            # qb = b / scale
            
            self.f = tf.Variable(f, dtype=tf.float32)
            self.g = tf.Variable(g, dtype=tf.float32)
            self.b = tf.Variable(b, dtype=tf.float32)
            
            # self.scale = tf.Variable(scale, dtype=tf.float32)
            
        else:
        
            f, b = weights[self.weight_id]['f'], weights[self.weight_id]['b']

            qf, scale = quantize_np(f, -128, 127)
            qb = b / scale
            
            self.f     = tf.Variable(qf,    dtype=tf.float32)
            self.b     = tf.Variable(qb,    dtype=tf.float32)
            self.scale = tf.Variable(scale, dtype=tf.float32)

    def train(self, x):
        x_pad = tf.pad(x, [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [0, 0]])
        '''
        conv = tf.nn.conv2d(x_pad, self.f, [1,1,1,1], 'VALID') # there is no bias when we have bn.
        mean = tf.reduce_mean(conv, axis=[0,1,2])
        _, var = tf.nn.moments(conv - mean, axes=[0,1,2])
        # mean, var = tf.nn.moments(conv, axes=[0,1,2])
        std = tf.sqrt(var + 1e-5)
        
        fold_f = (self.g * self.f) / std
        fold_b = self.b - ((self.g * mean) / std)
        qf = quantize_and_dequantize(fold_f, -128, 127)
        # qb = quantize_and_dequantize(fold_b, -128, 127) 
        '''
        
        conv = tf.nn.conv2d(x_pad, self.f, [1,self.p,self.p,1], 'VALID') + self.b
        
        if self.relu: out = tf.nn.relu(conv)
        else:         out = conv

        # out = quantize_and_dequantize(out, -128, 127)
        return out
        
    '''
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
        return qpool, [spool, std, mean]

    def predict(self, x, scale):
        # qf, sf = quantize(self.f, -128, 127)
        # qb = quantize_predict(self.b, sf, -2**24, 2**24-1)
        
        qf = self.f
        qb = self.b

        conv = tf.nn.conv2d(x, qf, [1,1,1,1], 'SAME') + qb
        relu = tf.nn.relu(conv)
        pool = tf.nn.avg_pool(relu, ksize=[1,self.p,self.p,1], strides=[1,self.p,self.p,1], padding='SAME')
        
        qpool = quantize_predict(pool, scale, -128, 127)

        # qpool = tf.Print(qpool, [self.layer_id, scale, tf.math.reduce_std(qf), tf.math.reduce_std(qb)], message='', summarize=1000)
        return qpool
        
    def inference(self, x):
        return self.predict(x=x, scale=self.q)
    '''
    
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.layer_id] = {'f': self.f, 'g': self.g, 'b': self.b}
        return weights_dict

    def get_params(self):
        return [self.f, self.b, self.g]

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
        
    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = tf.nn.relu(x + y2)
        return y3

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

    def train(self, x):
        y1 = self.conv1.train(x)
        y2 = self.conv2.train(y1)
        y3 = self.conv3.train(x)
        y4 = tf.nn.relu(y2 + y3)
        return y4
        
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

class dense_block(layer):
    def __init__(self, isize, osize, weights=None):
        self.weight_id = layer.weight_id
        layer.weight_id += 1
        self.layer_id = layer.layer_id
        layer.layer_id += 1
        
        self.isize = isize
        self.osize = osize
        
        w, b = weights[self.weight_id]['w'], weights[self.weight_id]['b']
        self.w = tf.Variable(w, dtype=tf.float32)
        self.b = tf.Variable(b, dtype=tf.float32)

    def train(self, x):
        x = tf.reshape(x, (-1, self.isize))
        fc = tf.matmul(x, self.w) + self.b
        return fc
        
    def get_weights(self):
        weights_dict = {}
        weights_dict[self.weight_id] = {'w': self.w, 'b': self.b}
        return weights_dict
        
    def get_params(self):
        return [self.w, self.b]

#############

class avg_pool(layer):
    def __init__(self, s, p, weights=None):
        self.layer_id = layer.layer_id
        layer.layer_id += 1
    
        self.s = s
        self.p = p
        
        if weights:
            self.q = weights['y']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_and_dequantize(pool, -128, 127)
        return qpool
    
    '''
    def collect(self, x):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return pool, [spool]

    def predict(self, x, scale):
        pool = tf.nn.avg_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool # quantize_predict(pool, scale, -128, 127) # this only works because we use np.ceil(scales)
        return qpool
        
    def inference(self, x):
        return self.predict(x=x, scale=self.q)
    '''
    
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
            self.q = weights['y']
        else:
            self.q = 1
        
    def train(self, x):        
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool 
        return qpool
    
    '''
    def collect(self, x):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool, spool = quantize(pool, -128, 127)
        return pool, [spool]

    def predict(self, x, scale):
        pool = tf.nn.max_pool(x, ksize=self.p, strides=self.s, padding="SAME")
        qpool = pool 
        return qpool
        
    def inference(self, x):
        return self.predict(x=x, scale=self.q)
    '''
    
    def get_weights(self):    
        weights_dict = {}
        weights_dict[self.layer_id] = {}
        return weights_dict

    def get_params(self):
        return []





        
        
        
