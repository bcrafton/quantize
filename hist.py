
import numpy as np
import matplotlib.pyplot as plt

weights = np.load('cifar10_weights.npy', allow_pickle=True).item()
for ii in range(6):
    x = np.reshape(weights[ii][0], -1)
    plt.hist(x, bins=100)
    plt.show()
