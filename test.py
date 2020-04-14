
import torch
import numpy as np
import torchvision
import torchvision.models.quantization as models

# You will need the number of filters in the `fc` for future use.
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model = models.resnet18(pretrained=True, progress=True, quantize=True)
num_ftrs = model.fc.in_features

##############################

# print (model)

##############################

def quantize_np(x, low, high):
  scale = (np.max(x) - np.min(x)) / (high - low)
  x = x / scale
  x = np.floor(x)
  x = np.clip(x, low, high)
  return x
  
'''
def dequantize(x, low, high):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Floor": "Identity"}):
        scale = (tf.reduce_max(x) - tf.reduce_min(x)) / (high - low)
        x = x * scale
        return x
'''

# this dosnt make any sense.
def dequantize_np(x, low, high):
  scale = (np.max(x) - np.min(x)) / (high - low)
  x = x * scale
  return x

##############################

f1 = model.fc.weight().int_repr().numpy()
b1 = model.fc.bias().detach().numpy()
z = model.fc.zero_point
s = model.fc.scale

##############################

f2 = model.fc.weight().dequantize().numpy()
b2 = model.fc.bias().detach().numpy()

##############################

x = np.random.randint(low=-128, high=127, size=512)

##############################

y_ref = f2 @ x + b2

##############################

scale = model.fc.weight().q_per_channel_scales().numpy().reshape(1000)
y = f1 @ x + b1

##############################

print (np.around(y / y_ref * scale))
# print (np.around(y))
























