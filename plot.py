
import numpy as np
import matplotlib.pyplot as plt 

acc = np.load('acc.npy', allow_pickle=True).item()
# sigmas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
sigmas = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]

###############################
'''
xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(0, 0, 'dynamic', sigma, 0.1)]['acc']
    ys.append(y)
plt.plot(xs, ys, marker='.')

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 0, 'dynamic', sigma, 0.1)]['acc']
    ys.append(y)
plt.plot(xs, ys, marker='.')

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 0.1)]['acc']
    ys.append(y)
plt.plot(xs, ys, marker='.')

###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 0.25)]['acc']
    ys.append(y)
plt.plot(xs, ys, marker='.')
'''
###############################

xs = sigmas
ys = []
for sigma in sigmas:
    y = acc[(1, 1, 'static', sigma, 0.1)]['acc']
    ys.append(y)
plt.plot(xs, ys, marker='.')

###############################

plt.show()












