# python test.py | grep "Tensor" -B 100

import numpy as np
x = np.load('trained_weights.npy', allow_pickle=True).item()
for l in x.keys():
    print ('#', l)
    for p in x[l].keys():
        print (p, type(x[l][p]))
