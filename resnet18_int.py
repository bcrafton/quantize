
import torch
import numpy as np
import torchvision
import torchvision.models.quantization as models

##############################

model = models.resnet18(pretrained=True, progress=True, quantize=True)

##############################

b = model.conv1.bias().detach().numpy()
s = model.conv1.scale
scale = model.conv1.weight().q_per_channel_scales().numpy() 
z = model.conv1.zero_point

##############################

wint = model.conv1.weight().int_repr().numpy().transpose(2, 3, 1, 0)
wint = np.reshape(wint, (7*7*3, 64))

###################################

wfloat = model.conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
wfloat = np.reshape(wfloat, (7*7*3, 64))

###################################

x = np.random.uniform(-1, 1, size=(7*7*3))
yfloat = x @ wfloat + b
yint = (x @ wint - z) * scale + b

print (yfloat[0:3])
print (yint[0:3])

###################################

x = x * 10 + 2

###################################

wfloat = wfloat / 10
bfloat = b - (2 * np.ones_like(x)) @ wfloat

yfloat = x @ wfloat + bfloat

print (yfloat[0:3])

###################################

print (wint.flatten()[0:5])
print ((wfloat / scale).flatten()[0:5])

###################################























