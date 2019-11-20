""" ResNet implementend as a torch Module.

Implementation following 'Deep Residual Learning for Image Recognition' by
He, Zhang, Ren, Sun https://arxiv.org/abs/1512.03385

To follow the 'official' pytroch ResNet implementation BatchNorm as well as
additional ReLUs in the ResNetBlocks were added:

    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

Author: Samuel Frommenwiler"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):

    def __init__(self,
                 blocks_per_layer,
                 num_classes,
                 num_channels,
                 initial_num_input_filters=64,
                 initial_num_output_filters=64,
                 bottleneck=False):
        super(ResNet, self).__init__()
        self.initial_num_input_filters = initial_num_input_filters
        self.initial_num_output_filters = initial_num_output_filters
        self.pooling_size = 1
        self.bottleneck = bottleneck
        self.trunk = nn.Sequential(
            nn.Conv2d(num_channels,
                      self.initial_num_input_filters,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.BatchNorm2d(self.initial_num_input_filters),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.blocks_per_layer = blocks_per_layer
        self.num_input_filters = initial_num_input_filters
        self.num_output_filters = initial_num_output_filters
        self.layers = nn.ModuleList()
        num_incoming_filters = self.initial_num_input_filters
        for layer_num, bpl in enumerate(blocks_per_layer):
            self.layers.append(self._build_layer(layer_num,
                                                 bpl,
                                                 num_incoming_filters,
                                                 bottleneck))
            num_incoming_filters = self.num_output_filters
            self.num_output_filters *= 2
            self.num_input_filters *= 2

        self.layers = nn.Sequential(*self.layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((self.pooling_size,
                                              self.pooling_size))
        self.fc_layer = nn.Sequential(
            nn.Linear(int(self.num_output_filters/2*self.pooling_size**2),
                      num_classes),
            nn.LogSoftmax(dim=1))


    def forward(self, x):
        x = self.trunk(x)
        x = self.layers(x)
        #print(x.shape)
        x = self.avg_pool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc_layer(x)
        return x

    def _build_layer(self,
                     layer_num,
                     blocks_per_layer,
                     num_incoming_filters,
                     bottleneck=False):
        layer = nn.ModuleList()
        downsample = False
        num_previous_output_filters = num_incoming_filters
        for block in range(blocks_per_layer):
            if(block == 0 and self.num_input_filters!=self.num_output_filters
              and num_incoming_filters!=self.num_input_filters):
                downsample = True
            print("{}_{}  #prev_fi: {}, #inp_fi: {}, #out_fi: {}, down: {}".format(
                layer_num, block, num_previous_output_filters, self.num_input_filters, self.num_output_filters, downsample))
            layer.append(ResNetBlock(num_previous_output_filters,
                                     self.num_input_filters,
                                     self.num_output_filters,
                                     downsample,
                                     bottleneck))
            if(block == 0 and self.num_input_filters!=self.num_output_filters):
                downsample = True
            downsample = False
            num_previous_output_filters = self.num_output_filters

        return nn.Sequential(*layer)

class ResNetBlock(nn.Module):

    def __init__(self,
                 num_incoming_filters,
                 num_input_filters,
                 num_output_filters,
                 downsample=False,
                 bottleneck=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        self.bottleneck = bottleneck
        first_stride = 1

        if(downsample):
            self.downsample_residual = nn.Sequential(
                nn.Conv2d(num_incoming_filters,
                                                 num_output_filters,
                                                 1,
                                                 stride=2,
                                                 padding=0,
                                                 bias=False),
                nn.BatchNorm2d(num_output_filters))
            first_stride = 2

        if(num_output_filters!=num_incoming_filters):#and downsample==False):
            self.downsample_residual = nn.Sequential(
                nn.Conv2d(num_incoming_filters,
                                                 num_output_filters,
                                                 1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=False),
                nn.BatchNorm2d(num_output_filters))
            self.downsample = True

        if(bottleneck):
            #print("#inc_fi: {}, #inp_fi: {}, #out_fi: {}".format(num_incoming_filters, num_input_filters, num_output_filters))
            self.block = nn.Sequential(
                nn.Conv2d(num_incoming_filters,
                          num_input_filters, 1,
                          padding=0,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(num_input_filters),
                nn.ReLU(),
                nn.Conv2d(num_input_filters, num_input_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_input_filters),
                nn.ReLU(),
                nn.Conv2d(num_input_filters, num_output_filters, 1, padding=0, bias=False),
                nn.BatchNorm2d(num_output_filters))
        else:
            self.block = nn.Sequential(
                nn.Conv2d(num_incoming_filters,
                          num_input_filters,
                          3,
                          padding=1,
                          stride=first_stride,
                          bias=False),
                nn.BatchNorm2d(num_input_filters),
                nn.ReLU(),
                nn.Conv2d(num_input_filters, num_output_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_output_filters))
        self.relu = nn.ReLU(inplace=True)
        #print(self.block)

    def forward(self, x):
        identity = x
        #print(x.shape)
        x = self.block(x)
        #print(x.shape)
        if self.downsample:
            identity = self.downsample_residual(identity)
        x += identity
        x = self.relu(x)
        return x

