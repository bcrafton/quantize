
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

def dequantize_np(x, low, high):
  scale = (np.max(x) - np.min(x)) / (high - low)
  x = x * scale
  return x

##############################

f1 = model.conv1.weight().int_repr().numpy().transpose(2, 3, 1, 0)
b1 = model.conv1.bias().detach().numpy()
z = model.conv1.zero_point
s = model.conv1.scale

##############################

f2 = model.conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
b2 = model.conv1.bias().detach().numpy()

##############################

print (f1[:, :, 0, 0] / f2[:, :, 0, 0])

print (f1[:, :, 0, 0])
print ()
print (f2[:, :, 0, 0])
print ()




