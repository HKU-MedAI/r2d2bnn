import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import HalfCauchy, Dirichlet, Exponential
from distributions import ReparametrizedGaussian, ScaleMixtureGaussian,\
    InverseGamma, Exponential, Gamma, InvGaussian, GeneralizedInvGaussian
from scipy.special import loggamma
from losses import calculate_kl


class R2D2LinearLayer(nn.Module):
    """
    Single linear layer of a R2D2 prior for regression
    """
    def __init__(self, in_features, out_features, parameters):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class R2D2 Hyperparameters
            device: cuda device instance
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.priors = parameters

        # Scale to initialize weights, according to Yingzhen's work
        if parameters["r2d2_scale"] == None:
            scale = 1. * np.sqrt(6. / (in_features + out_features))
        else:
            scale = parameters["r2d2_scale"]

        # Initialization of parameters of variational distribution
        # weight parameters
        self.tot_dim = out_features * in_features
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-scale, scale))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]))
        self.beta_ = ReparametrizedGaussian(self.beta_mean, self.beta_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([out_features]))
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # Initialization of parameters online
        # Initialization of distributions local shrinkage parameters
        # weight parameters
        self.prior_psi_shape = parameters["prior_psi_shape"]
        self.psi_ = Exponential(torch.ones(out_features, in_features) * self.prior_psi_shape)
        # Distribution of Phi_
        self.a_pi = parameters["prior_phi_prob"]
        self.phi_ = Dirichlet(torch.ones(out_features, in_features) * self.a_pi)

        # Initialization of Global shrinkage parameters
        # Distribution of Xi
        self.prior_xi_shape = torch.Tensor([parameters["weight_xi_shape"]])
        self.prior_xi_rate = torch.Tensor([1])
        self.xi_ = Gamma(self.prior_xi_shape, self.prior_xi_rate)

        # Distribution of Omega
        self.prior_omega_rate = torch.Tensor([parameters["weight_omega_shape"]])
        self.xi = self.xi_.sample().squeeze()
        self.omega_ = Gamma(self.tot_dim * self.a_pi, self.xi)
        self.psi = self.psi_.sample()
        self.phi = self.phi_.sample()
        self.omega = self.omega_.sample().squeeze()
        self.beta = self.beta_.sample()

        # Initialization of distributions for Gibbs sampling
        self.xi_gib = Gamma(self.a_pi + self.prior_xi_shape, 1 + self.omega)
        self.omega_gib = GeneralizedInvGaussian(
            chi=2 * torch.sum(self.beta ** 2 / (self.beta_.std_dev ** 2 * self.phi * self.psi)),
            rho=2 * self.xi,
            lamb=(self.a_pi - 1 / 2) * self.tot_dim
        )
        self.t_gib = GeneralizedInvGaussian(
            chi=2 * self.beta ** 2 / (self.beta_.std_dev ** 2 * self.phi * self.psi),
            rho=2 * self.xi,
            lamb=self.a_pi - 1 / 2
        )
        self.psi_gib = GeneralizedInvGaussian(
            chi=-1 / 2 * torch.ones(1),
            rho=1 / torch.sqrt(self.beta_.std_dev ** 2 + self.phi * self.omega / 2) / torch.abs(self.beta),
            lamb=torch.ones(1)
        )

        # Define prior quantities for calculating the KL loss
        # TODO: Temporary solutions using Gaussian distribution - develop analytic form later
        self.prior_mu = 0
        self.prior_beta_sigma = torch.sqrt(self.phi * self.psi * self.omega * self.beta_.std_dev ** 2 / 2).detach()
        self.prior_bias_sigma = self.bias.std_dev.detach()

        self.reset_parameters()

    def reset_parameters(self):
        # self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.beta_rho.data.normal_(*self.priors["beta_rho_scale"])

        # self.bias_mu.data.normal_(*self.posterior_mu_initial)
        self.bias_rho.data.normal_(*self.priors["bias_rho_scale"])


    def forward(self, input_, sample=True, n_samples=10):
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
        beta = self.beta_.sample(n_samples)
        beta_eps = torch.empty(self.beta.size()).normal_(0, 1)
        beta_sigma = torch.sqrt(self.beta_.std_dev ** 2 * self.omega * self.phi * self.psi / 2)

        weight = beta + beta_sigma * beta_eps

        bias = self.bias.sample(n_samples)

        input_ = input_.expand(n_samples, -1, -1)

        input_ = input_.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

        self.beta = beta.squeeze()  # Update beta

        result = torch.einsum('bij,bkj->bik', [input_, weight]) + bias.unsqueeze(1).expand(-1, input_.shape[1], -1)
        return result

    def analytic_update(self):
        """
        Calculates analytic updates of sample, gamma
        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        # Refer to Gibbs sampling algorithm for marginal R2D2 https://arxiv.org/pdf/1609.00046.pdf

        # Sample phi from InverseGaussian
        beta = self.beta_.mean.detach()
        beta_sigma = self.beta_.std_dev.detach()

        self.psi_gib.update(
            chi=torch.ones(1),
            rho=beta ** 2 / (beta_sigma ** 2 * self.phi * self.omega / 2),
            lamb=-1 / 2 * torch.ones(1)
        )
        self.psi = self.psi_gib.sample().squeeze(0) ** -1
        self.psi[self.psi == 0] += 1e-8  # Ensure non-zero

        # Update omega distribution and Sample omega
        self.omega_gib.update(
            chi=torch.sum(2 * beta ** 2 / (beta_sigma ** 2 * self.phi * self.psi)),
            rho=2 * self.xi,
            lamb=(self.a_pi - 1 / 2) * self.tot_dim
        )
        self.omega = self.omega_gib.sample()

        # Update full posterior of xi and sample xi
        self.xi_gib.update(self.a_pi * self.tot_dim + self.prior_xi_shape, 1 + self.omega)
        self.xi = self.xi_gib.sample().squeeze()

        # Sample phi
        self.t_gib.update(
            chi=2 * beta ** 2 / (beta_sigma ** 2 * self.psi),
            rho=2 * self.xi,
            lamb=self.a_pi - 1 / 2
        )
        t = self.t_gib.sample()
        self.phi = t / torch.sum(t)
        self.phi[self.phi == 0] += 1e-8

    def kl_loss(self):
        beta_sigma = self.beta_.std_dev.detach()
        bias_sigma = self.bias.std_dev.detach()
        kl = calculate_kl(self.prior_mu, self.prior_beta_sigma, self.beta, beta_sigma)
        kl += calculate_kl(self.prior_mu, self.prior_bias_sigma, self.bias_mean, bias_sigma)
        return kl

