import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from radialbnn.distributions import Gaussian


class RadialLayer(nn.Module):

    """Radial Linear Layer.

    Implementation of a Radial Linear Layer as described in 'Radial
    Bayesian Neural Networks: Beyond Discrete Support In Large-Scale
    Bayesian Deep Learning'.
    """

    def __init__(self, in_features: int, out_features: int) -> None:

        """Radial Linear Layer.

        Parameters
        ----------
        in_features : int
            Number of features to feed into the layer.
        out_features : int
            Number of features produced by the layer.
        """

        super().__init__()

        w_mu = torch.empty(out_features, in_features).uniform_(-0.2, 0.2)
        w_rho = torch.empty(out_features, in_features).uniform_(-5.0, -4.0)

        bias_mu = torch.empty(out_features).uniform_(-0.2, 0.2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)

        self.w_mu = nn.Parameter(w_mu)
        self.w_rho = nn.Parameter(w_rho)

        self.bias_mu = nn.Parameter(bias_mu)
        self.bias_rho = nn.Parameter(bias_rho)

        self.prior = Gaussian(0, 1)
        self.epsilon_normal = torch.distributions.Normal(0, 0.5)

        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass through the linear layer.

        Parameters
        ----------
        x : Tensor
            Inputs to the Radial Linear Layer.

        Returns
        -------
        Tensor
            Output from the Radial Linear Layer.
        """

        device = self.w_mu.device

        # calculating sigma from rho
        w_std = torch.log(1 + torch.exp(self.w_rho))
        bias_std = torch.log(1 + torch.exp(self.bias_rho))

        # draw weight from radial distribution
        w_eps_mfvi = self.epsilon_normal.sample(self.w_mu.size()).to(device)
        w_eps_norm = torch.norm(w_eps_mfvi, p=2, dim=0)

        w_eps_normalised = w_eps_mfvi / w_eps_norm
        w_r_mfvi = torch.randn(1).to(device)

        w_eps = w_eps_normalised * w_r_mfvi

        # draw bias from radial distribution
        bias_eps_mfvi = self.epsilon_normal.sample(self.bias_mu.size()).to(device)
        bias_eps_norm = torch.norm(bias_eps_mfvi, p=2, dim=0)

        bias_eps_normalised = bias_eps_mfvi / bias_eps_norm
        bias_r_mfvi = torch.randn(1).to(device)

        bias_eps = bias_eps_normalised * bias_r_mfvi

        # calculate weight and bias
        w = torch.addcmul(self.w_mu, w_std, w_eps)
        bias = torch.addcmul(self.bias_mu, bias_std, bias_eps)

        # calculate log probabilities
        w_log_posterior = -torch.sum(torch.log(w_std))
        bias_log_posterior = -torch.sum(torch.log(bias_std))

        w_log_prior = self.prior.log_likelihood(w)
        bias_log_prior = self.prior.log_likelihood(bias)

        total_log_posterior = w_log_posterior + bias_log_posterior
        total_log_prior = w_log_prior + bias_log_prior

        # calculate kl divergence
        self.kl_divergence = total_log_posterior - total_log_prior

        return F.linear(x, w, bias)