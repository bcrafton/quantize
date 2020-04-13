
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

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
    # output = torch.nn.functional.softmax(output[0], dim=0)

image = input_tensor.detach().numpy().transpose(1,2,0)
np.save('input_image', image)

output = output.detach().numpy()
output = output[0]
# print (image[0,0,:])
print (np.argmax(output))
print (output[np.argmax(output)])
print (output)

# import matplotlib.pyplot as plt
# plt.imshow(image)
# plt.show()

# print (dir(model.layer1))
# print (dir(model.layer1[0].conv1))
# print (model.layer1[0].conv1.features)


x = input_batch
count = 0
for layer in model.children():
    x = layer(x)
    a = np.transpose(x.detach().numpy(), (0, 2, 3, 1))
    count += 1
    if (count == 1): break

np.save('ref', a)

import matplotlib.pyplot as plt
a = a[0, :, :, 1]
print (np.shape(a))
print (a[0])
plt.imshow(a)
plt.show()








