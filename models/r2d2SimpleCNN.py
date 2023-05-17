import math
import torch.nn as nn

from layers import (
    R2D2LinearLayer,
    R2D2ConvLayer,
    FlattenLayer,
)


class R2D2SimpleCNN(nn.Module):
    def __init__(self, outputs, inputs, priors, image_size=32, activation_type='softplus'):
        super().__init__()

        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus()
        elif activation_type == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Only softplus or relu supported")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = R2D2ConvLayer(inputs, 16, priors, 16, padding=2)
        out_size = (image_size - 16 + 2 * 2) // 1 + 1
        out_size = (out_size - 2) // 2 + 1

        self.flatten = FlattenLayer(out_size * out_size * 16)
        self.fc1 = R2D2LinearLayer(out_size * out_size * 16, outputs, self.priors)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)

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

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        for module in self.children():
            if hasattr(module, 'analytic_update'):
                module.analytic_update()

        return None

    def get_map(self, x):
        return self.conv1(x)