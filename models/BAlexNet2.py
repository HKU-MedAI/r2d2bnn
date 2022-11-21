import torch
import torch.nn as nn
from torch.nn import functional as F
from layers import (
    BBBConv2d,
    BBBLinear,
    FlattenLayer
)


class BBBAlexNet2(nn.Module):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBAlexNet2, self).__init__()

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
                BBBConv2d(inputs, 64, 11, stride=4, padding=2, bias=True, priors=self.priors),
                BBBConv2d(64, 192, 5, padding=2, bias=True, priors=self.priors),
                BBBConv2d(192, 384, 3, padding=1, bias=True, priors=self.priors),
                BBBConv2d(384, 256, 3, padding=1, bias=True, priors=self.priors),
                BBBConv2d(256, 256, 3, padding=1, bias=True, priors=self.priors)
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
                FlattenLayer(8 * 8 * 256)
            )
        )

        self.linears_prediction = nn.ModuleList(
            (
                BBBLinear(31 * 31 * 64, 256, bias=True, priors=self.priors),
                BBBLinear(15 * 15 * 192, 256, bias=True, priors=self.priors),
                BBBLinear(8 * 8 * 256, 256, bias=True, priors=self.priors)
            )
        )
        self.classifier = BBBLinear(256, outputs, bias=True, priors=self.priors)

    def forward(self, x):

        emb_list = []

        x = self.convs[0](x)
        x = self.act(x)
        x = self.pools[0](x)
        e = self.linears_prediction[0](self.flattens[0](x))
        emb_list.append(e)

        x = self.convs[1](x)
        x = self.act(x)
        x = self.pools[1](x)
        e = self.linears_prediction[1](self.flattens[1](x))
        emb_list.append(e)

        x = self.convs[2](x)
        x = self.act(x)

        x = self.convs[3](x)
        x = self.act(x)

        x = self.convs[4](x)
        x = self.act(x)
        x = self.pools[2](x)
        e = self.linears_prediction[2](self.flattens[2](x))
        emb_list.append(e)

        x = self.classifier(e)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        # Get final embeddings
        embs = torch.stack(emb_list)
        embs = F.normalize(embs, dim=2)
        embs = embs.mean(0)

        return x, kl, embs

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
