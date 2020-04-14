
import torch
import torchvision.models.quantization as models

model = models.resnet18(pretrained=True, progress=True, quantize=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet152', pretrained=True)
model.eval()

from PIL import Image
from torchvision import transforms
import numpy as np

input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

scale = (torch.max(input_batch) - torch.min(input_batch)) / 255.
# input_batch = torch.floor(input_batch * scale)
# https://github.com/pytorch/pytorch/issues/28070

output = model(input_batch)

# x = torch.quantize_per_tensor(input_batch, scale=scale, zero_point=0, dtype=torch.qint8)
x = torch.quantize_per_tensor(input_batch, scale=scale, zero_point=128, dtype=torch.quint8)

qx = (x.int_repr().numpy())

####################################

count = 0
for i, layer in enumerate(model.children()):
    # print (i, layer)
    if i == 9:
        x = torch.reshape(x, (1, 512))
    if i == 10:
        break

    x = layer(x)
    
####################################

print (x.shape)
print (torch.argmax(x.int_repr()))

'''
y1 = model.conv1(x)
y2 = model.maxpool(y1)
y3 = model.layer1[0].conv1(y2)
y4 = model.layer1[0].conv2(y3)
'''

'''
import matplotlib.pyplot as plt
# y = y.cpu().detach().numpy()
y = y[0, 0, :, :]
plt.imshow(y)
plt.show()
'''

####################################




