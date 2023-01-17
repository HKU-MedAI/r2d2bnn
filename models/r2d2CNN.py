import math
import torch.nn as nn

from layers import (
    R2D2LinearLayer,
    R2D2ConvLayer,
    FlattenLayer,
)


class BBBR2D2CNN(nn.Module):
    '''The architecture of LeNet with Bayesian r2d2 Layers'''

    def __init__(self, outputs, inputs, priors, n_blocks, image_size=32, activation_type='softplus'):
        super(BBBR2D2CNN, self).__init__()

        self.num_classes = outputs
        self.priors = priors
        self.n_blocks = n_blocks

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        convs = [
            R2D2ConvLayer(inputs, 32, priors, 5, padding=2),
            R2D2ConvLayer(32, 64, priors, 5, padding=2),
            R2D2ConvLayer(64, 128, priors, 5, padding=1),
            R2D2ConvLayer(128, 128, priors, 2, padding=1),
        ]

        pools = [
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]

        self.conv_block = nn.Sequential()

        out_size = image_size
        out_channels = 0

        for l in range(self.n_blocks):
            self.conv_block.add_module(f"conv{l}", convs[l])
            self.conv_block.add_module(f"act{l}", self.act())
            self.conv_block.add_module(f"pool{l}", pools[l])
            out_size = (out_size - 5 + 2 * convs[l].padding) // 1 + 1
            out_size = (out_size - 3) // 2 + 1
            out_channels = convs[l].out_features

        self.dense_block = nn.Sequential(
            FlattenLayer(out_size * out_size * out_channels),
            R2D2LinearLayer(out_size * out_size * out_channels, 1000, self.priors),
            self.act(),
            R2D2LinearLayer(1000, 1000, self.priors),
            self.act(),
            R2D2LinearLayer(1000, outputs, self.priors),
        )

    def kl_loss(self):
        """
        Calculates the kl loss of each layer
        """
        kl = 0
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'kl_loss'):
                    kl += cm.kl_loss()

        return kl

    def analytic_update(self):
        """
        Calculates the update of the model parameters with
        analytic update equations
        """
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'analytic_update'):
                    cm.analytic_update()

        return None

    def forward(self, x):

        x = self.conv_block(x)
        x = self.dense_block(x)

        return x
