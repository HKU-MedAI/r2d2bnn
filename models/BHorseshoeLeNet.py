import math
import torch.nn as nn

from layers import (
    HorseshoeConvLayer,
    HorseshoeLinearLayer,
    FlattenLayer,
)


class BBBHorseshoeLeNet(nn.Module):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, image_size=32, activation_type='softplus'):
        super(BBBHorseshoeLeNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = HorseshoeConvLayer(inputs, 6, priors, 5, padding=0)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (image_size - 5 + 1) // 2

        self.conv2 = HorseshoeConvLayer(6, 16, priors, 5, padding=0)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        out_size = (out_size - 5 + 1) // 2

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = HorseshoeLinearLayer(out_size * out_size * 16, 120, self.priors)
        self.act3 = self.act()

        self.fc2 = HorseshoeLinearLayer(120, 84, self.priors)
        self.act4 = self.act()

        self.fc3 = HorseshoeLinearLayer(84, outputs, self.priors)

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

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return kl

    def forward(self, x):

        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)

        x = self.fc2(x)
        x = self.act4(x)

        x = self.fc3(x)

        return x
