
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

print (model)

##############################

# https://discuss.pytorch.org/t/how-to-extract-individual-weights-after-per-channel-static-quantization/67456/2

##############################

# print (model.conv1.weight().int_repr())
# print (model.conv1.weight().dequantize())

##############################

weights = np.array(model.conv1.weight().int_repr())
# print (np.max(weights))
# print (np.min(weights))

##############################

weight_dict = {}

for layer in range(21):
    weight_dict[layer] = {}

##############################

weight_dict[0]['f'] = model.conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)

weight_dict[1]['f'] = model.layer1[0].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[2]['f'] = model.layer1[0].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[3]['f'] = model.layer1[1].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[4]['f'] = model.layer1[1].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)

weight_dict[5]['f'] = model.layer2[0].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[6]['f'] = model.layer2[0].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[7]['f'] = model.layer2[0].downsample[0].weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[8]['f'] = model.layer2[1].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[9]['f'] = model.layer2[1].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)

weight_dict[10]['f'] = model.layer3[0].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[11]['f'] = model.layer3[0].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[12]['f'] = model.layer3[0].downsample[0].weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[13]['f'] = model.layer3[1].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[14]['f'] = model.layer3[1].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)

weight_dict[15]['f'] = model.layer4[0].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[16]['f'] = model.layer4[0].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[17]['f'] = model.layer4[0].downsample[0].weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[18]['f'] = model.layer4[1].conv1.weight().dequantize().numpy().transpose(2, 3, 1, 0)
weight_dict[19]['f'] = model.layer4[1].conv2.weight().dequantize().numpy().transpose(2, 3, 1, 0)

###################################

# print (dir(model.conv1.bias()))
# print (model.conv1.weight().int_repr())
# print (model.conv1.bias().int_repr())
# print (model.conv1.bias.numpy())
# print (model.conv1.weight())
# print (model.conv1.bias())
# print (model.conv1.bias().detach().numpy())

weight_dict[0]['b'] = model.conv1.bias().detach().numpy()

weight_dict[1]['b'] = model.layer1[0].conv1.bias().detach().numpy()
weight_dict[2]['b'] = model.layer1[0].conv2.bias().detach().numpy()
weight_dict[3]['b'] = model.layer1[1].conv1.bias().detach().numpy()
weight_dict[4]['b'] = model.layer1[1].conv2.bias().detach().numpy()

weight_dict[5]['b'] = model.layer2[0].conv1.bias().detach().numpy()
weight_dict[6]['b'] = model.layer2[0].conv2.bias().detach().numpy()
weight_dict[7]['b'] = model.layer2[0].downsample[0].bias().detach().numpy()
weight_dict[8]['b'] = model.layer2[1].conv1.bias().detach().numpy()
weight_dict[9]['b'] = model.layer2[1].conv2.bias().detach().numpy()

weight_dict[10]['b'] = model.layer3[0].conv1.bias().detach().numpy()
weight_dict[11]['b'] = model.layer3[0].conv2.bias().detach().numpy()
weight_dict[12]['b'] = model.layer3[0].downsample[0].bias().detach().numpy()
weight_dict[13]['b'] = model.layer3[1].conv1.bias().detach().numpy()
weight_dict[14]['b'] = model.layer3[1].conv2.bias().detach().numpy()

weight_dict[15]['b'] = model.layer4[0].conv1.bias().detach().numpy()
weight_dict[16]['b'] = model.layer4[0].conv2.bias().detach().numpy()
weight_dict[17]['b'] = model.layer4[0].downsample[0].bias().detach().numpy()
weight_dict[18]['b'] = model.layer4[1].conv1.bias().detach().numpy()
weight_dict[19]['b'] = model.layer4[1].conv2.bias().detach().numpy()

###################################

weight_dict[0]['s'] = model.conv1.scale

weight_dict[1]['s'] = model.layer1[0].conv1.scale          
weight_dict[2]['s'] = model.layer1[0].conv2.scale          
weight_dict[3]['s'] = model.layer1[1].conv1.scale          
weight_dict[4]['s'] = model.layer1[1].conv2.scale          

weight_dict[5]['s'] = model.layer2[0].conv1.scale          
weight_dict[6]['s'] = model.layer2[0].conv2.scale          
weight_dict[7]['s'] = model.layer2[0].downsample[0].scale 
weight_dict[8]['s'] = model.layer2[1].conv1.scale          
weight_dict[9]['s'] = model.layer2[1].conv2.scale         

weight_dict[10]['s'] = model.layer3[0].conv1.scale         
weight_dict[11]['s'] = model.layer3[0].conv2.scale        
weight_dict[12]['s'] = model.layer3[0].downsample[0].scale 
weight_dict[13]['s'] = model.layer3[1].conv1.scale        
weight_dict[14]['s'] = model.layer3[1].conv2.scale        

weight_dict[15]['s'] = model.layer4[0].conv1.scale         
weight_dict[16]['s'] = model.layer4[0].conv2.scale         
weight_dict[17]['s'] = model.layer4[0].downsample[0].scale 
weight_dict[18]['s'] = model.layer4[1].conv1.scale         
weight_dict[19]['s'] = model.layer4[1].conv2.scale       

###################################

weight_dict[0]['z'] = model.conv1.zero_point

weight_dict[1]['z'] = model.layer1[0].conv1.zero_point
weight_dict[2]['z'] = model.layer1[0].conv2.zero_point
weight_dict[3]['z'] = model.layer1[1].conv1.zero_point
weight_dict[4]['z'] = model.layer1[1].conv2.zero_point

weight_dict[5]['z'] = model.layer2[0].conv1.zero_point
weight_dict[6]['z'] = model.layer2[0].conv2.zero_point
weight_dict[7]['z'] = model.layer2[0].downsample[0].zero_point
weight_dict[8]['z'] = model.layer2[1].conv1.zero_point
weight_dict[9]['z'] = model.layer2[1].conv2.zero_point

weight_dict[10]['z'] = model.layer3[0].conv1.zero_point
weight_dict[11]['z'] = model.layer3[0].conv2.zero_point
weight_dict[12]['z'] = model.layer3[0].downsample[0].zero_point
weight_dict[13]['z'] = model.layer3[1].conv1.zero_point
weight_dict[14]['z'] = model.layer3[1].conv2.zero_point

weight_dict[15]['z'] = model.layer4[0].conv1.zero_point
weight_dict[16]['z'] = model.layer4[0].conv2.zero_point
weight_dict[17]['z'] = model.layer4[0].downsample[0].zero_point
weight_dict[18]['z'] = model.layer4[1].conv1.zero_point
weight_dict[19]['z'] = model.layer4[1].conv2.zero_point

###################################

weight_dict[0]['scale'] = model.conv1.weight().q_per_channel_scales().numpy() 

weight_dict[1]['scale'] = model.layer1[0].conv1.weight().q_per_channel_scales().numpy() 
weight_dict[2]['scale'] = model.layer1[0].conv2.weight().q_per_channel_scales().numpy()
weight_dict[3]['scale'] = model.layer1[1].conv1.weight().q_per_channel_scales().numpy()
weight_dict[4]['scale'] = model.layer1[1].conv2.weight().q_per_channel_scales().numpy()

weight_dict[5]['scale'] = model.layer2[0].conv1.weight().q_per_channel_scales().numpy()
weight_dict[6]['scale'] = model.layer2[0].conv2.weight().q_per_channel_scales().numpy()
weight_dict[7]['scale'] = model.layer2[0].downsample[0].weight().q_per_channel_scales().numpy()
weight_dict[8]['scale'] = model.layer2[1].conv1.weight().q_per_channel_scales().numpy()
weight_dict[9]['scale'] = model.layer2[1].conv2.weight().q_per_channel_scales().numpy()

weight_dict[10]['scale'] = model.layer3[0].conv1.weight().q_per_channel_scales().numpy()
weight_dict[11]['scale'] = model.layer3[0].conv2.weight().q_per_channel_scales().numpy()
weight_dict[12]['scale'] = model.layer3[0].downsample[0].weight().q_per_channel_scales().numpy()
weight_dict[13]['scale'] = model.layer3[1].conv1.weight().q_per_channel_scales().numpy()
weight_dict[14]['scale'] = model.layer3[1].conv2.weight().q_per_channel_scales().numpy()

weight_dict[15]['scale'] = model.layer4[0].conv1.weight().q_per_channel_scales().numpy()
weight_dict[16]['scale'] = model.layer4[0].conv2.weight().q_per_channel_scales().numpy()
weight_dict[17]['scale'] = model.layer4[0].downsample[0].weight().q_per_channel_scales().numpy()
weight_dict[18]['scale'] = model.layer4[1].conv1.weight().q_per_channel_scales().numpy()
weight_dict[19]['scale'] = model.layer4[1].conv2.weight().q_per_channel_scales().numpy()

###################################

weight_dict[20]['w'] = model.fc.weight().dequantize().numpy().transpose(1, 0)
weight_dict[20]['b'] = model.fc.bias().detach().numpy()
weight_dict[20]['s'] = model.fc.scale 
weight_dict[20]['scale'] = model.fc.weight().q_per_channel_scales().numpy()
weight_dict[20]['z'] = model.fc.zero_point 

###################################

np.save('resnet18_quant', weight_dict)











