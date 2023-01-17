import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import (
    HorseshoeLinearLayer,
    HorseshoeConvLayer,
    FlattenLayer
)


class BBBHorseshoeAlexNet(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBHorseshoeAlexNet, self).__init__()

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
                HorseshoeConvLayer(inputs, 64, self.priors, 11, stride=4, padding=2),
                HorseshoeConvLayer(64, 192, self.priors, 5, padding=2),
                HorseshoeConvLayer(192, 384, self.priors, 3, padding=1),
                HorseshoeConvLayer(384, 256, self.priors, 3, padding=1),
                HorseshoeConvLayer(256, 256, self.priors, 3, padding=1)
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

        self.classifier = HorseshoeLinearLayer(256, outputs, priors=self.priors)

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

    def log_prior(self):
        """
        Calculates the logarithm of the current
        value of the prior distribution over the weights
        """
        lp = 0

        for module in self.children():
            if hasattr(module, 'log_prior'):
                lp += module.log_prior()

        return lp

    def log_variational_posterior(self):
        """
        Calculates the logarithm of the current value
        of the variational posterior distribution over the weights
        """
        lvp = 0
        for module in self.children():
            if hasattr(module, 'log_variational_posterior'):
                lvp += module.log_variational_posterior()

        return lvp

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        for module in self.children():
            if hasattr(module, 'analytic_update'):
                module.analytic_update()

        return None

    def inference(self, x):
        """

        :param x: Data
        :return:
        """
        maps = []
        maps.append(x.cpu().numpy())
        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)
        maps.append(x.cpu().numpy())

        x = self.convs[1](x)
        x = self.act(x)
        maps.append(x.cpu().numpy())
        x = self.pools[1](x)

        x = self.convs[2](x)
        x = self.act(x)
        maps.append(x.cpu().numpy())

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)

        return maps
