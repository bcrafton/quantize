
import numpy as np

###############

x = np.random.normal(size=1000)
w = np.random.normal(size=(1000, 1000))
b = np.random.normal(size=1000)

y1 = w @ x + b

###############

y2 = w @ (1000 * x) + b

###############

y3 = w @ (1000 * x) + (1000 * b)

###############

print (y3 / y1)
print (y2 / y1)
