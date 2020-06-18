
import torch
import torchvision
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models.quantization as models
import time
from load import Loader

###############################################################

# model = models.resnet18(pretrained=True, progress=True, quantize=True)
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
# this part is essential !?
model.eval()

###############################################################

load = Loader('/home/brian/Desktop/ILSVRC2012/val')

total = 50000
batch_size = 50

accum_correct = 0
accum = 0

start = time.time()
for batch in range(0, total, batch_size):
    while load.empty(): pass
    
    x, y = load.pop()
    with torch.no_grad():
        
        x = x.transpose(0,3,1,2)
        output = model(torch.from_numpy(x)).detach().numpy()
        pred = np.argmax(output, axis=1)

        correct = np.sum(y == pred)
        accum_correct += correct
        accum += batch_size

        if (accum % 100) == 0:
            print (accum / (time.time() - start), accum, accum_correct / accum)

load.join()
        

            
            
            
            
            
            
