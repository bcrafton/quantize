
import torch
import numpy as np

model = torch.hub.load('pytorch/vision:v0.5.0', 'resnet18', pretrained=True)
print (model)

# we shud be fine to switch to cuda 10.1 if need be.
# we not using any tensorflow code anymore.

weights = []
for param in model.parameters():
    weight = param.detach().numpy()
    weights.append(weight)
    # print (np.shape(weight))

###################################

# print ('-----')
# print (len(weights))

# weight_table = [[None for p in range(3)] for l in range(17)] 

weight_dict = {}
for layer in range(20):
    f = np.transpose(weights[layer * 3 + 0], (2, 3, 1, 0))
    g = weights[layer * 3 + 1]
    b = weights[layer * 3 + 2]
    weight_dict[layer] = {'f': f, 'g': g, 'b': b}
    print (layer, np.shape(f))

w = np.transpose(weights[60], (1, 0))
b = weights[61]
weight_dict[20] = {'w': w, 'b': b}
print (np.shape(w))

# np.save('resnet18', weight_dict)

###################################

# print (dir(model))
# print (model.layer1)
# print (model.layer1[0])
# print (model.layer1[0].conv1)

###################################

# print (model.bn1.running_var)

###################################

# print (model.layer1[0].bn1.running_var)
# print (model.layer1[0].bn2.running_var)

# print (model.layer1[1].bn1.running_var)
# print (model.layer1[1].bn2.running_var)

###################################

# print (model.layer2[0].bn1.running_var)
# print (model.layer2[0].bn2.running_var)
# print (model.layer2[0].downsample[1].running_var)

# print (model.layer2[1].bn1.running_var)
# print (model.layer2[1].bn2.running_var)

###################################

# print (model.layer3[0].bn1.running_var)
# print (model.layer3[0].bn2.running_var)
# print (model.layer3[0].downsample[1].running_var)

# print (model.layer3[1].bn1.running_var)
# print (model.layer3[1].bn2.running_var)

###################################

# print (model.layer4[0].bn1.running_var)
# print (model.layer4[0].bn2.running_var)
# print (model.layer4[0].downsample[1].running_var)

# print (model.layer4[1].bn1.running_var)
# print (model.layer4[1].bn2.running_var)


###################################

weight_dict[0]['mean'] = model.bn1.running_mean.detach().numpy()

weight_dict[1]['mean'] = model.layer1[0].bn1.running_mean.detach().numpy()
weight_dict[2]['mean'] = model.layer1[0].bn2.running_mean.detach().numpy()
weight_dict[3]['mean'] = model.layer1[1].bn1.running_mean.detach().numpy()
weight_dict[4]['mean'] = model.layer1[1].bn2.running_mean.detach().numpy()

weight_dict[5]['mean'] = model.layer2[0].bn1.running_mean.detach().numpy()
weight_dict[6]['mean'] = model.layer2[0].bn2.running_mean.detach().numpy()
weight_dict[7]['mean'] = model.layer2[0].downsample[1].running_mean.detach().numpy()
weight_dict[8]['mean'] = model.layer2[1].bn1.running_mean.detach().numpy()
weight_dict[9]['mean'] = model.layer2[1].bn2.running_mean.detach().numpy()

weight_dict[10]['mean'] = model.layer3[0].bn1.running_mean.detach().numpy()
weight_dict[11]['mean'] = model.layer3[0].bn2.running_mean.detach().numpy()
weight_dict[12]['mean'] = model.layer3[0].downsample[1].running_mean.detach().numpy()
weight_dict[13]['mean'] = model.layer3[1].bn1.running_mean.detach().numpy()
weight_dict[14]['mean'] = model.layer3[1].bn2.running_mean.detach().numpy()

weight_dict[15]['mean'] = model.layer4[0].bn1.running_mean.detach().numpy()
weight_dict[16]['mean'] = model.layer4[0].bn2.running_mean.detach().numpy()
weight_dict[17]['mean'] = model.layer4[0].downsample[1].running_mean.detach().numpy()
weight_dict[18]['mean'] = model.layer4[1].bn1.running_mean.detach().numpy()
weight_dict[19]['mean'] = model.layer4[1].bn2.running_mean.detach().numpy()

###################################

weight_dict[0]['var'] = model.bn1.running_var.detach().numpy()

weight_dict[1]['var'] = model.layer1[0].bn1.running_var.detach().numpy()
weight_dict[2]['var'] = model.layer1[0].bn2.running_var.detach().numpy()
weight_dict[3]['var'] = model.layer1[1].bn1.running_var.detach().numpy()
weight_dict[4]['var'] = model.layer1[1].bn2.running_var.detach().numpy()

weight_dict[5]['var'] = model.layer2[0].bn1.running_var.detach().numpy()
weight_dict[6]['var'] = model.layer2[0].bn2.running_var.detach().numpy()
weight_dict[7]['var'] = model.layer2[0].downsample[1].running_var.detach().numpy()
weight_dict[8]['var'] = model.layer2[1].bn1.running_var.detach().numpy()
weight_dict[9]['var'] = model.layer2[1].bn2.running_var.detach().numpy()

weight_dict[10]['var'] = model.layer3[0].bn1.running_var.detach().numpy()
weight_dict[11]['var'] = model.layer3[0].bn2.running_var.detach().numpy()
weight_dict[12]['var'] = model.layer3[0].downsample[1].running_var.detach().numpy()
weight_dict[13]['var'] = model.layer3[1].bn1.running_var.detach().numpy()
weight_dict[14]['var'] = model.layer3[1].bn2.running_var.detach().numpy()

weight_dict[15]['var'] = model.layer4[0].bn1.running_var.detach().numpy()
weight_dict[16]['var'] = model.layer4[0].bn2.running_var.detach().numpy()
weight_dict[17]['var'] = model.layer4[0].downsample[1].running_var.detach().numpy()
weight_dict[18]['var'] = model.layer4[1].bn1.running_var.detach().numpy()
weight_dict[19]['var'] = model.layer4[1].bn2.running_var.detach().numpy()

###################################

weight_dict[0]['f'] = model.conv1.weight.detach().numpy().transpose(2, 3, 1, 0)

weight_dict[1]['f'] = model.layer1[0].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[2]['f'] = model.layer1[0].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[3]['f'] = model.layer1[1].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[4]['f'] = model.layer1[1].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)

weight_dict[5]['f'] = model.layer2[0].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[6]['f'] = model.layer2[0].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[7]['f'] = model.layer2[0].downsample[0].weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[8]['f'] = model.layer2[1].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[9]['f'] = model.layer2[1].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)

weight_dict[10]['f'] = model.layer3[0].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[11]['f'] = model.layer3[0].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[12]['f'] = model.layer3[0].downsample[0].weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[13]['f'] = model.layer3[1].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[14]['f'] = model.layer3[1].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)

weight_dict[15]['f'] = model.layer4[0].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[16]['f'] = model.layer4[0].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[17]['f'] = model.layer4[0].downsample[0].weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[18]['f'] = model.layer4[1].conv1.weight.detach().numpy().transpose(2, 3, 1, 0)
weight_dict[19]['f'] = model.layer4[1].conv2.weight.detach().numpy().transpose(2, 3, 1, 0)

###################################

weight_dict[0]['g'] = model.bn1.weight.detach().numpy()

weight_dict[1]['g'] = model.layer1[0].bn1.weight.detach().numpy()
weight_dict[2]['g'] = model.layer1[0].bn2.weight.detach().numpy()
weight_dict[3]['g'] = model.layer1[1].bn1.weight.detach().numpy()
weight_dict[4]['g'] = model.layer1[1].bn2.weight.detach().numpy()

weight_dict[5]['g'] = model.layer2[0].bn1.weight.detach().numpy()
weight_dict[6]['g'] = model.layer2[0].bn2.weight.detach().numpy()
weight_dict[7]['g'] = model.layer2[0].downsample[1].weight.detach().numpy()
weight_dict[8]['g'] = model.layer2[1].bn1.weight.detach().numpy()
weight_dict[9]['g'] = model.layer2[1].bn2.weight.detach().numpy()

weight_dict[10]['g'] = model.layer3[0].bn1.weight.detach().numpy()
weight_dict[11]['g'] = model.layer3[0].bn2.weight.detach().numpy()
weight_dict[12]['g'] = model.layer3[0].downsample[1].weight.detach().numpy()
weight_dict[13]['g'] = model.layer3[1].bn1.weight.detach().numpy()
weight_dict[14]['g'] = model.layer3[1].bn2.weight.detach().numpy()

weight_dict[15]['g'] = model.layer4[0].bn1.weight.detach().numpy()
weight_dict[16]['g'] = model.layer4[0].bn2.weight.detach().numpy()
weight_dict[17]['g'] = model.layer4[0].downsample[1].weight.detach().numpy()
weight_dict[18]['g'] = model.layer4[1].bn1.weight.detach().numpy()
weight_dict[19]['g'] = model.layer4[1].bn2.weight.detach().numpy()

###################################

weight_dict[0]['b'] = model.bn1.bias.detach().numpy()

weight_dict[1]['b'] = model.layer1[0].bn1.bias.detach().numpy()
weight_dict[2]['b'] = model.layer1[0].bn2.bias.detach().numpy()
weight_dict[3]['b'] = model.layer1[1].bn1.bias.detach().numpy()
weight_dict[4]['b'] = model.layer1[1].bn2.bias.detach().numpy()

weight_dict[5]['b'] = model.layer2[0].bn1.bias.detach().numpy()
weight_dict[6]['b'] = model.layer2[0].bn2.bias.detach().numpy()
weight_dict[7]['b'] = model.layer2[0].downsample[1].bias.detach().numpy()
weight_dict[8]['b'] = model.layer2[1].bn1.bias.detach().numpy()
weight_dict[9]['b'] = model.layer2[1].bn2.bias.detach().numpy()

weight_dict[10]['b'] = model.layer3[0].bn1.bias.detach().numpy()
weight_dict[11]['b'] = model.layer3[0].bn2.bias.detach().numpy()
weight_dict[12]['b'] = model.layer3[0].downsample[1].bias.detach().numpy()
weight_dict[13]['b'] = model.layer3[1].bn1.bias.detach().numpy()
weight_dict[14]['b'] = model.layer3[1].bn2.bias.detach().numpy()

weight_dict[15]['b'] = model.layer4[0].bn1.bias.detach().numpy()
weight_dict[16]['b'] = model.layer4[0].bn2.bias.detach().numpy()
weight_dict[17]['b'] = model.layer4[0].downsample[1].bias.detach().numpy()
weight_dict[18]['b'] = model.layer4[1].bn1.bias.detach().numpy()
weight_dict[19]['b'] = model.layer4[1].bn2.bias.detach().numpy()

###################################

np.save('resnet18', weight_dict)

###################################

'''
print (dir(model.layer4[0].bn1))
print (model.layer4[0].bn1.weight)
# print (model.layer4[0].bn1.eps)
# print (model.layer4[0].bn1.parameters())
for param in model.layer4[0].bn1.parameters():
    print (param.detach().numpy())

'''














