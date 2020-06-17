
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

from load import Loader
load = Loader('/home/brian/Desktop/ILSVRC2012/val')

total_correct = 0
total = 0

while load.empty(): pass
while not load.empty():
    x, y = load.pop()
    with torch.no_grad():
        x = x.transpose(0,3,1,2)
        output = model(torch.from_numpy(x)).detach().numpy()
        pred = np.argmax(output, axis=1)

        correct = np.sum(y == pred)
        total_correct += correct
        total += 50

        if (total % 1000) == 0:
            print (total_correct / total)
        

            
            
            
            
            
            
