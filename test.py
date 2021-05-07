
import numpy as np

w = np.load('cifar10_weights.npy', allow_pickle=True).item()

for l in w.keys():
    if 'f' in w[l].keys():
        f = w[l]['f']
        print (np.shape(f), np.max(f), np.min(f))
    if 'w' in w[l].keys():
        w = w[l]['w']
        print (np.shape(w), np.max(w), np.min(w))
