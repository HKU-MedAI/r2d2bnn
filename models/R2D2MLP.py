import torch.nn as nn

from layers import (
    R2D2LinearLayer
)


class R2D2MultipleLinear(nn.Module):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """

    def __init__(self, outputs, inputs, priors, n_blocks=3, layer_type="r2d2", activation_type='softplus'):
        super(R2D2MultipleLinear, self).__init__()

        self.num_classes = outputs
        self.priors = priors

        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.n_blocks = n_blocks

        linears = [
                R2D2LinearLayer(inputs, 32, self.priors),
                R2D2LinearLayer(32, 64, self.priors),
                R2D2LinearLayer(64, 128, self.priors),
                R2D2LinearLayer(128, 128, self.priors)
        ]

        out_channel = inputs

        self.dense_block = nn.Sequential()

        for l in range(self.n_blocks):
            self.dense_block.add_module(f"fc{l}", linears[l])
            self.dense_block.add_module(f"act{l}", self.act())
            out_channel = linears[l].out_features

        fc_out = R2D2LinearLayer(out_channel, outputs, self.priors)
        self.dense_block.add_module(f"fc_out", fc_out)

    def kl_loss(self):
        # Compute KL divergences
        kl = 0.0
        for module in self.children():
            for cm in module.children():
                if hasattr(cm, 'kl_loss'):
                    kl = kl + cm.kl_loss()

        return kl

    def forward(self, x):
        x = self.dense_block(x)
        return x

