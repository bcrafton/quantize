
import torch
model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
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

# scale = 255. / (torch.max(input_batch) - torch.min(input_batch))
# input_batch = torch.floor(input_batch * scale)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
    # output = torch.nn.functional.softmax(output[0], dim=0)

print (torch.argmax(output))

x = input_batch
for layer in model.children():
    y = layer(x)
    break
    
import matplotlib.pyplot as plt
y = y.cpu().detach().numpy()
y = y[0, 0, :, :]
plt.imshow(y)
plt.show()








