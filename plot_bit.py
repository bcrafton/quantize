
import numpy as np
import matplotlib.pyplot as plt

def count_ones(x):
    count = 0
    for bit in range(8):
        count += np.bitwise_and(np.right_shift(x, bit), 1)
    return count

x = []
y = []
for val in range(256):
    x.append(val)
    y.append(count_ones(val))
    
plt.plot(x, y)
plt.show()
