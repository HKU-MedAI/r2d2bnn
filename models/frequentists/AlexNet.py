import torch.nn as nn
from layers import (
    FlattenLayer
)


class AlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, activation_type='softplus'):
        super(AlexNet, self).__init__()

        self.num_classes = outputs

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = nn.Conv2d(inputs, 64, 11, stride=4, padding=2, bias=True)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 5, padding=2, bias=True)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, 3, padding=1, bias=True)
        self.act3 = self.act()

        self.conv4 = nn.Conv2d(384, 256, 3, padding=1, bias=True)
        self.act4 = self.act()

        self.conv5 = nn.Conv2d(256, 256, 3, padding=1, bias=True)
        self.act5 = self.act()
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.flatten = FlattenLayer(1 * 1 * 256)
        self.classifier = nn.Linear(1 * 1 * 256, outputs, bias=True)
