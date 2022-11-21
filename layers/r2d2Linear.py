import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import HalfCauchy, Dirichlet, Exponential
from distributions import ReparametrizedGaussian, ScaleMixtureGaussian,\
    InverseGamma, Exponential, Gamma, InvGaussian, GeneralizedInvGaussian
from scipy.special import loggamma


class R2D2Layer(nn.Module):
    """
    Single linear layer of a R2D2 prior for regression
    """
    def __init__(self, in_features, out_features, parameters, device):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class HorseshoeHyperparameters
            device: cuda device instance
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # Scale to initialize weights, according to Yingzhen's work
        if parameters.horseshoe_scale == None:
            scale = 1. * np.sqrt(6. / (in_features + out_features))
        else:
            scale = parameters.horseshoe_scale

        # TODO: Maybe refer the the Refer to the Horseshoe BNN part on how to compute the theoretical ELBO

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-scale, scale))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * parameters.beta_rho_scale)
        self.beta_ = ReparametrizedGaussian(self.beta_mean, self.beta_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([1, out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * parameters.bias_rho_scale)
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # Initialization of parameters online
        # Initialization of distributions local shrinkage parameters
        # weight parameters
        self.prior_psi_shape = torch.Tensor([0.5])
        self.psi_ = Exponential(self.prior_psi_shape)
        # Distribution of Phi_
        self.a_pi = torch.Tensor([1])
        self.phi_ = Dirichlet(self.a_pi)

        # Initialization of Global shrinkage parameters
        # Distribution of Xi
        self.prior_xi_shape = torch.Tensor([parameters.weight_xi_shape])
        self.prior_xi_rate = torch.Tensor([1])
        self.xi_ = Gamma(self.prior_xi_shape, self.prior_xi_rate)

        # Distribution of Omega
        self.prior_omega_rate = torch.Tensor([parameters.weight_omega_shape])
        xi = self.xi_.sample(1)
        self.omega_ = Gamma(self.prior_omega_rate, xi)
        self.psi = self.psi_.sample()
        self.phi = self.phi_.sample()
        self.omega = self.omega_.sample()
        self.xi = self.xi_.sample()
        self.beta = self.beta_.sample()

        # Initialization of distributions for Gibbs sampling

    def forward(self, input_, sample=True, n_samples=1):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.
        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """

        # Compute variance parameter
        # It is phi_j and psi_j for local shrinkage
        phi = torch.unsqueeze(self.phi_.sample(n_samples), 1)
        psi = torch.unsqueeze(self.psi_.sample(n_samples), 1)
        xi = self.xi_.sample()
        self.omega_.update(self.prior_omega_rate, xi)
        omega = torch.unsqueeze(self.omega_.sample(n_samples), 1)
        self.beta_sigma = torch.log1p(torch.exp(self.beta_rho)) * omega * xi / 2 * psi * phi

        weight = self.beta_mean + self.beta_sigma

        bias = self.bias.sample(n_samples)

        input_ = input_.expand(n_samples, -1, -1)

        if self.device.type == 'cuda':
            input_ = input_.cuda()
            weight = weight.cuda()
            bias = bias.cuda()

        result = torch.einsum('bij,bkj->bik', [input_, weight]) + bias
        return result

    def analytic_update(self):
        """
        Calculates analytic updates of sample, gamma
        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        # TODO Update each of the parameters here using Gibbs sampling
        # Refer to Gibbs sampling algorithm for marginal R2D2 https://arxiv.org/pdf/1609.00046.pdf

        # Sample phi from InverseGaussian
        self.psi = self.psi_.sample()

        # Sample omega
        chi = torch.sum(2 * self.beta / (self.beta_sigma ** 2 * self.phi * self.psi))
        r = 2 * self.xi
        lamb_0 = self.prior_xi_shape - self.out_features / 2
        omega = self.omega_.sample()  # TODO: Create a GIG from this

        # Sample xi from gamma

        # Sample phi

        # TODO: Update the distributions of shrinkage parameters???
        # TODO: Priors and gibbs marginal posterior, which one to take?
