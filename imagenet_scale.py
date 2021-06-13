
import numpy as np
import tensorflow as tf
from layers import *
from load import Loader
import time

####################################

train_flag = False
weights = np.load('trained_weights.npy', allow_pickle=True).item()

####################################

model = model(layers=[
conv_block((7,7,3,64), 2, weights=weights, train=train_flag),

max_pool(2, 3),

res_block1(64,   64, 1, weights=weights, train=train_flag),
res_block1(64,   64, 1, weights=weights, train=train_flag),

res_block2(64,   128, 2, weights=weights, train=train_flag),
res_block1(128,  128, 1, weights=weights, train=train_flag),

res_block2(128,  256, 2, weights=weights, train=train_flag),
res_block1(256,  256, 1, weights=weights, train=train_flag),

res_block2(256,  512, 2, weights=weights, train=train_flag),
res_block1(512,  512, 1, weights=weights, train=train_flag),

avg_pool(7, 7),
dense_block(512, 1000, weights=weights, train=train_flag)
])

####################################

w = model.save_weights()
np.save('resnet18_weights.npy', w)

####################################

print (w.keys())
for key in w.keys():
    if ('q' in w[key].keys()) and ('sf' in w[key].keys()):
        print (w[key]['q'])



