import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import (
    R2D2ConvLayer,
    R2D2LinearLayer,
    FlattenLayer
)


class BBBR2D2AlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBR2D2AlexNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")

        self.convs = nn.ModuleList(
            (
                R2D2ConvLayer(inputs, 64, self.priors, 11, stride=4, padding=2),
                R2D2ConvLayer(64, 192, self.priors, 5, padding=2),
                R2D2ConvLayer(192, 384, self.priors, 3, padding=1),
                R2D2ConvLayer(384, 256, self.priors, 3, padding=1),
                R2D2ConvLayer(256, 256, self.priors, 3, padding=1)
            )
        )

        self.pools = nn.ModuleList(
            (
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=1, stride=2)
            )
        )

        self.flattens = nn.ModuleList(
            (
                FlattenLayer(31 * 31 * 64),
                FlattenLayer(15 * 15 * 192),
                FlattenLayer(1 * 1 * 256)
            )
        )

        self.classifier = R2D2LinearLayer(1 * 1 * 256, outputs, self.priors)

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        for module in self.children():
            if hasattr(module, 'analytic_update'):
                module.analytic_update()

        return None

    def forward(self, x):

        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)

        x = self.convs[1](x)
        x = self.act(x)
        x = self.pools[1](x)

        x = self.convs[2](x)
        x = self.act(x)

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)

        x = self.flattens[2](x)

        x = self.classifier(x)

        return x

    def kl_loss(self):
        """
        Calculates the kl loss of each layer
        """
        kl = 0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl += module.kl_loss()

        return kl
