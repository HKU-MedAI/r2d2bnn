import math
import torch.nn as nn
import torch

from layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer,
    ReverseFlattenLayer,
    ResizeLayer
)


class BBBConvNet(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBConvNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        # self.conv1 = nn.Conv2d(inputs, 6, 5, 1, 0, bias=True)
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        # self.conv2 = nn.Conv2d(6, 16, 5, 1, 0, bias=True)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv1 = BBBConv2d(inputs, 16, 3, 2, padding=1, bias=True, priors=self.priors)
        # self.act1 = nn.LeakyReLU(0.1)

        # self.conv2 = BBBConv2d(16, 32, 3, 2, padding=1, bias=True, priors=self.priors)
        # self.act2 = nn.LeakyReLU(0.1)

        # self.conv3 = BBBConv2d(32, 64, 3, 2, padding=1, bias=True, priors=self.priors)
        # self.act3 = nn.LeakyReLU(0.1)

        # self.conv4 = BBBConv2d(64, 128, 3, 2, padding=1, bias=True, priors=self.priors)
        # self.act4 = nn.LeakyReLU(0.1)

        # self.conv5 = BBBConv2d(128, outputs, 3, 2, padding=1, bias=True, priors=self.priors)
        # self.act5 = nn.Tanh


        # self.conv1 = nn.Conv2d(inputs, 16, 3, 2, 1, bias=True)
        # self.act1 = nn.LeakyReLU(0.1)
        # self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, bias=True)
        # self.act2 = nn.LeakyReLU(0.1)
        # self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, bias=True)
        # self.act3 = nn.LeakyReLU(0.1)
        # self.conv4 = nn.Conv2d(64, 128, 3, 2, 1, bias=True)
        # self.act4 = nn.LeakyReLU(0.1)
        # self.conv5 = nn.Conv2d(128, outputs, 3, 2, 1, bias=True)
        # self.act5 = nn.Tanh

        # self.conv1 = nn.Conv2d(inputs, 32, 3, 2, 0, bias=True)
        # self.act1 = nn.LeakyReLU(0.1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=True)
        # self.act2 = nn.LeakyReLU(0.1)
        # self.conv3 = nn.Conv2d(64, 128, 3, 1, 0, bias=True)
        # self.act3 = nn.LeakyReLU(0.1)
        # self.conv4 = nn.Conv2d(128, 128, 3, 1, 0, bias=True)
        # self.act4 = nn.LeakyReLU(0.1)
        # self.conv5 = nn.Conv2d(128, 64, 3, 1, 0, bias=True)
        # self.act5 = nn.LeakyReLU(0.1)
        # self.conv6 = nn.Conv2d(64, 32, 3, 1, 0, bias=True)
        # self.act6 = nn.LeakyReLU(0.1)
        # self.conv7 = nn.Conv2d(32, 32, 3, 1, 0, bias=True)
        # self.act7 = nn.LeakyReLU(0.1)
        # self.conv8 = nn.Conv2d(32, outputs, 3, 1, 0, bias=True)
        # self.act8 = nn.Tanh

class BBBClassifier(nn.Module):
    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBClassifier, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        
        self.fc1 = BBBLinear(inputs, 120, bias=True, priors=self.priors)
        # self.fc1 = nn.Linear(inputs, 120, bias=True)
        self.act6 = nn.LeakyReLU(0.1)

        self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
        # self.fc2 = nn.Linear(120, 84, bias=True)
        self.act7 = nn.LeakyReLU(0.1)

        # self.fc3 = nn.Linear(84, outputs, bias=True)
        self.fc3 = BBBLinear(84, outputs, bias=True, priors=self.priors)

        
        # self.fc1 = nn.Linear(inputs, 128, bias=True)
        # self.act6 = nn.LeakyReLU(0.1)
        # self.fc2 = nn.Linear(128, 128, bias=True)
        # self.act7 = nn.LeakyReLU(0.1)
        # self.fc3 = nn.Linear(128, outputs, bias=True)
        
    
    def forward(self, x1, x2=None):
        b, c, w, h = x1.shape
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)

        x = x1 - x2
        # x = torch.cat([x1,x2], 1)

        for module in self.children():
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()
        return x, kl

        

