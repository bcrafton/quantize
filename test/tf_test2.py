
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
#########################

results = np.load('results.npy', allow_pickle=True).item()
x = results['x'].transpose(0,2,3,1).astype(np.float32)
qx = results['qx'].transpose(0,2,3,1).astype(np.float32)
y_ref = results['y'].transpose(0,2,3,1)

weights = np.load('iweights.npy', allow_pickle=True).item()
f = weights[0]['f']
b = weights[0]['b']
s = weights[0]['s']
z = weights[0]['z']
scale = weights[0]['scale']

def conv2d(x, f, b, pad, stride):
    x_pad = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    conv = tf.nn.conv2d(x_pad, f, [1,stride,stride,1], 'VALID') * scale + b
    conv = conv / s
    return conv

#########################

y = conv2d(x, f, b, 3, 2)

print (np.max(y_ref))
print (tf.reduce_max(y))
print ()
print (y_ref[0][0][0][0])
print (y[0][0][0][0])
print ()

#########################

x_pad = np.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='constant')
t = x_pad[0, 0:7, 0:7, :]
t = np.reshape(t, (7,7,3,1))

y = (f * t) 
y = np.sum(y, axis=(0,1,2))
y = y * scale + b
y = y / s

print ()

#########################

print (y[0])

#########################
