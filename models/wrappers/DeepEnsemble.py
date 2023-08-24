from torch import nn

from parse import parse_model


class DeepEnsemble(nn.Module):
    def __init__(self, config):
        super(DeepEnsemble, self).__init__()

        self.n_model = config["n_models"]
        self.models = nn.ModuleList(
            [parse_model(config) for _ in self.n_model]
        )

    def forward(self):
        pass
