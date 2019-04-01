# -*- coding: utf-8 -*-
"""
Created on 2018-12-04 10:23:06

@author: AN Zhulin
"""

# TODO: add connection form input to the first num_dense_layer layers in NEAT!
import sys,os
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath+"/neat-cnn")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

shuffle = True

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)

class cnn_block(nn.Module):

    layerout = list()
    downsampling_time = list()
    num_dense_layer = 0

    '''Depthwise conv + Pointwise conv'''
    def __init__(self, downsampling_time, num_dense_layer, in_planes, out_planes, stride=1):
        super(cnn_block, self).__init__()

        global shuffle

        if shuffle:
            g = in_planes // out_planes

            if g == 0:
                g = 1
        else:
            g = 1

        if shuffle:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, g*out_planes, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn2 = nn.BatchNorm2d(g*out_planes)
            self.conv2 = nn.Conv2d(g*out_planes, g*out_planes, kernel_size=3, stride=stride, padding=1, groups=g, bias=True)
            self.shuffle = ShuffleBlock(groups=g)
            self.bn3 = nn.BatchNorm2d(g*out_planes)
            self.conv3 = nn.Conv2d(g*out_planes, out_planes, kernel_size=1, stride=1, padding=0, groups=out_planes, bias=True)
            self.num_dense_layer = num_dense_layer
            self.downsampling_time = downsampling_time
        else:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, g * out_planes, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn2 = nn.BatchNorm2d(g * out_planes)
            self.conv2 = nn.Conv2d(g * out_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=out_planes, bias=True)
            self.num_dense_layer = num_dense_layer
            self.downsampling_time = downsampling_time

        #self.dropout = nn.Dropout2d(p=0.1)

    def clear_layerout(self):
        self.layerout.clear()

    def forward(self, x):

        layer_num = len(self.layerout)

        if layer_num == 0:

            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))

            self.layerout.append(x)
            self.layerout.append(out)

            return out
        else:

            former_layers = list()
            former_layers_downsampling_time = list()

            # Save the former num_dense_layer layers with its downsampling tiems
            if layer_num <= self.num_dense_layer:
                former_layers.append(self.layerout[0])
                former_layers_downsampling_time.append(0)
                for i in range(1, layer_num):
                    former_layers.append(self.layerout[i])
                    former_layers_downsampling_time.append(self.downsampling_time[i-1])
            else:
                for i in range(self.num_dense_layer):
                    former_layers.append(self.layerout[layer_num-self.num_dense_layer+i])
                    former_layers_downsampling_time.append(self.downsampling_time[layer_num-self.num_dense_layer+i-1])

            # Computer the feature map downsampling times
            for i in range(len(former_layers_downsampling_time)):
                former_layers_downsampling_time[i] = former_layers_downsampling_time[-1] - former_layers_downsampling_time[i]
                while former_layers_downsampling_time[i] > 0:
                    former_layers[i] = F.avg_pool2d(former_layers[i], 2)
                    former_layers_downsampling_time[i] -= 1

            # Concatnate the input
            if layer_num <= self.num_dense_layer:
                input = former_layers[0]
                for i in range(1, len(former_layers_downsampling_time)):
                    input = torch.cat([input, former_layers[i]], 1)
            else:
                input = former_layers[0]
                for i in range(1, self.num_dense_layer):
                    input = torch.cat([input, former_layers[i]], 1)

            if shuffle:
                out = self.conv1(F.relu(self.bn1(input)))
                out = self.conv2(F.relu(self.bn2(out)))
                out = self.shuffle(out)
                out = self.conv3(F.relu(self.bn3(out)))
            else:
                out = self.conv1(F.relu(self.bn1(input)))
                out = self.conv2(F.relu(self.bn2(out)))

            self.layerout.append(out)

            return out

class fc_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(in_planes, out_planes)
        self.bn = nn.BatchNorm1d(out_planes)

        #self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        #out = F.relu(self.bn(self.dropout(self.fc(x))))
        out = F.relu(self.bn(self.fc(x)))
        #out = self.fc(x)
        return out

class Net(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    # cfg = [32, 32, (64, 2), 64, (128,2), 128, (256,2), 256, (512,2)]

    def __init__(self, config):
        super(Net, self).__init__()

        # Save useful neat parameters

        self.num_cnn_layer = config.num_cnn_layer
        self.num_layer = config.num_layer
        self.num_inputs = config.num_inputs
        self.num_outputs = config.num_outputs
        self.input_size = config.input_size
        self.num_downsampling = config.num_downsampling

        self.downsampling_mask = []
        self.downsampling_time = []
        self.nodes_every_layers = []

        #self.downsampling_mask = genome.downsampling_mask
        #self.downsampling_time = genome.downsampling_time
        #self.nodes_every_layers = genome.nodes_every_layers
        self.num_dense_layer = config.num_dense_layer

        #self.out = list()
        self.setup_node_list(config)

        cfg = self.setup_cfg()

        self.cnn_layers = self._make_cnn_layers(cfg)
        self.fc_layers = self._make_fc_layers()
        #self.clear_parameters() #Note! Should not clear parameters!

        # if set_parameters:
            # self.set_parameters(genome)

    def setup_cfg(self):
        cfg = list()

        for i in range(self.num_layer):
            if self.downsampling_mask[i]:
                cfg.append((self.nodes_every_layers[i], 2))
            else:
                cfg.append(self.nodes_every_layers[i])
        return cfg

    def setup_node_list(self, config):
        # Compute node number in every layer.
        assert config.num_cnn_layer >= config.num_downsampling

        if config.num_cnn_layer == config.num_downsampling:
            layer_num_in_middle_group = 1
            layer_num_in_last_group = 1
            layer_num_in_first_group = 1
            group_num = config.num_downsampling
            self.downsampling_mask = [0] * config.num_layer
            for i in range(config.num_cnn_layer):
                self.downsampling_mask[i] = 1
        else:
            # The layers are arranged in groups.
            # The middle groups have num_cnn_layer // num_downsampling layers
            # The first and last group have (num_cnn_layer // num_downsampling) + (num_cnn_layer % num_downsampling) layers.
            # The first group have one more layer than last group if the total layer is not even.
            layer_num_in_middle_group = config.num_cnn_layer // config.num_downsampling
            layer_num_in_last_group = (layer_num_in_middle_group + (
                    config.num_cnn_layer % config.num_downsampling)) // 2
            layer_num_in_first_group = (layer_num_in_middle_group + (
                    config.num_cnn_layer % config.num_downsampling)) - layer_num_in_last_group
            group_num = config.num_downsampling + 1

            # Generate downsampling mask.
            self.downsampling_mask = [0] * config.num_layer
            for i in range(layer_num_in_first_group, (config.num_cnn_layer - (layer_num_in_last_group - 1))):
                if (i - layer_num_in_first_group) % layer_num_in_middle_group == 0:
                    self.downsampling_mask[i] = 1
                else:
                    self.downsampling_mask[i] = 0

            # Convert downsampling mask to downsampling num list
            times = 0
            self.downsampling_time = [0] * config.num_layer
            for i in range(config.num_layer):
                if self.downsampling_mask[i] == 1:
                    times += 1
                self.downsampling_time[i] = times

            # Generate node number in each layer.
            # self.nodes_every_layers.append(config.num_inputs)
            for i in range(layer_num_in_first_group):
                self.nodes_every_layers.append(config.init_channel_num)

            for i in range(1, group_num - 1):
                for j in range(layer_num_in_middle_group):
                    self.nodes_every_layers.append(config.init_channel_num * (2 ** i))

            # Note "layer_num_in_last_group - 1"
            # for i in range(self.layer_num_in_last_group - 1):
            for i in range(layer_num_in_last_group):
                self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))

            # Generate node number in fc layers.
            # Note the last layer have fixed node num equals to output num.
            for i in range(config.num_cnn_layer, config.num_layer - 1):
                self.nodes_every_layers.append(config.init_channel_num * (2 ** config.num_downsampling))

            # Add the last fc layer node number.
            self.nodes_every_layers.append(config.num_outputs)

    def _make_cnn_layers(self, cfg):

        layers = []

        for i in range(self.num_cnn_layer):
            if i < self.num_dense_layer:
                in_planes = self.num_inputs
                for j in range(i):
                    in_planes += self.nodes_every_layers[j]
            else:
                in_planes = 0
                for j in range(self.num_dense_layer):
                    in_planes += self.nodes_every_layers[i-j-1]

            x = cfg[i]
            out_planes = self.nodes_every_layers[i]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(cnn_block(self.downsampling_time, self.num_dense_layer, in_planes, out_planes, stride))
        return nn.Sequential(*layers)

    def _make_fc_layers(self):
        layers = []
        in_planes = 0
        for i in range(self.num_dense_layer):
            in_planes += self.nodes_every_layers[self.num_cnn_layer-1-i]

        layers.append(fc_block(in_planes, self.nodes_every_layers[self.num_cnn_layer]))

        for i in range(self.num_cnn_layer + 1, self.num_layer):
            layers.append(fc_block(self.nodes_every_layers[i-1], self.nodes_every_layers[i]))
        return nn.Sequential(*layers)

    # Note! Should not clear!
    # set all weight and bias to zero
    """
    def clear_parameters(self):
        for module in self.children():
            for block in module:
                if isinstance(block, Block):
                    block.conv1.weight.data.fill_(0.0)
                    # block.conv1.bias.fill_(0.0)
                    block.conv2.weight.data.fill_(0.0)
                    # block.conv2.bias.fill_(0.0)
                else:
                    block.weight.data.fill_(0.0)
                    block.bias.data.fill_(0.0)
    """
    def forward(self, x):
        cnn_block.clear_layerout(cnn_block)
        out = self.cnn_layers(x)

        former_layers = list()
        former_layers_downsampling_time = list()

        # Save the former num_dense_layer layers with its downsampling tiems

        for i in range(self.num_dense_layer):
            former_layers.append(cnn_block.layerout[self.num_cnn_layer - self.num_dense_layer+1+i])
            former_layers_downsampling_time.append(self.downsampling_time[self.num_cnn_layer - self.num_dense_layer + i])

        # Computer the feature map downsampling times
        for i in range(len(former_layers_downsampling_time)):
            former_layers_downsampling_time[i] = former_layers_downsampling_time[-1] - \
                                                 former_layers_downsampling_time[i]
            while former_layers_downsampling_time[i] > 0:
                former_layers[i] = F.avg_pool2d(former_layers[i], 2)
                former_layers_downsampling_time[i] -= 1

        # Concatnate the input
        input = former_layers[0]
        for i in range(1, self.num_dense_layer):
            input = torch.cat([input, former_layers[i]], 1)

        out = F.avg_pool2d(input, self.input_size // (2 ** self.num_downsampling))
        out = out.view(out.size(0), -1)
        out = self.fc_layers(out)
        return out
