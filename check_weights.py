
import numpy as np

x = np.load('resnet18_quant_weights.npy', allow_pickle=True).item()
for key in x.keys():
    if 'y' in x[key].keys():
        print (key, x[key]['y'])





