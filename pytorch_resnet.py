
import torch
import torchvision
import numpy as np
import os

from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt

import torchvision.models.quantization as models

# model = models.resnet18(pretrained=True, progress=True, quantize=True)
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# this part is essential !?
model.eval()

# imagenet = torchvision.datasets.ImageNet(root='/home/brian/Desktop/ILSVRC2012', split='val')
# load = torch.utils.data.DataLoader(imagenet, batch_size=4, shuffle=True, num_workers=args.nThreads)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

correct = 0
total = 0
pred = np.zeros(shape=50000)

path = '/home/brian/Desktop/ILSVRC2012/val'
for subdir, dirs, files in os.walk(path):
    for file in files:
        label = int(subdir.split('/')[-1])
        
        filename = subdir + '/' + file
        input_image = Image.open(filename).convert('RGB')
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch).detach().numpy().flatten()
            pred[total] = np.argmax(output)
            correct += np.argmax(output) == label
            total += 1
        
        if (total % 1000 == 0):
            print (correct, total, correct / total)

print (correct, total, correct / total)

# Before '.convert('RGB')' : 0.6844131245035541
# After:
# Expected: 69.8

# Quantized : 0.68132
# Expected: 69.4
            
# np.save('predictions', pred)
                       
# images are named the same !
# we have 1:50k
# we just cant search between the two because JPEG vs jpg.
            
            
            
            
            
            
