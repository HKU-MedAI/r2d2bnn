
import numpy as np
import torch
import math
from abc import ABCMeta, abstractmethod
from scipy.stats import norm, bernoulli
from scipy.special import gamma, digamma, loggamma, logsumexp
import random


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f


def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f


def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return


def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    p = float(p); a = float(a); b = float(b)
    lam = p
    omega = math.sqrt(a*b)

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0/(alpha+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0/(alpha+2.0*lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0/(alpha*math.cosh(1)+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        elif lam == 0:
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha)))

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0/xi
    r = 1.0/zeta

    td = t-r*eta
    sd = s-p*theta
    q = td+sd

    # random variate generation
    # TODO: Can modify this into multi dimension
    while True:
        U = random.random()
        V = random.random()
        W = random.random()
        if U < q/(p+q+r):
            rnd = -sd+q*V
        elif U < (q+r)/(p+q+r):
            rnd = td-r*math.log(V)
        else:
            rnd = -sd+p*math.log(V)

        f1 = math.exp(-eta-zeta*(rnd-t))
        f2 = math.exp(-theta+xi*(rnd+s))
        if W*g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+math.pow(lam,2)/math.pow(omega,2)))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd


class ReparametrizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """
    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        return self.mean + self.std_dev * epsilon

    def logprob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                    - torch.log(self.std_dev)
                    - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum()

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            # n_inputs, n_outputs = self.mean.shape
            dim = 1
            for d in self.mean.shape:
                dim *= d
        else:
            dim = len(self.mean)
            # n_outputs = 1

        part1 = dim / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return part1 + part2

class ScaleMixtureGaussian(Distribution):
    """
    Scale Mixture of two Gaussian distributions with zero mean but different
    variances.
    """
    def __init__(self, mixing_coefficient, sigma1, sigma2):
        torch.manual_seed(42)
        self.mixing_coefficient = mixing_coefficient
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def logprob(self, target):
        if self.mixing_coefficient == 1.0:
            prob = self.gaussian1.log_prob(target)
            logprob = prob.sum()
        else:
            prob1 = torch.exp(self.gaussian1.log_prob(target))
            prob2 = torch.exp(self.gaussian2.log_prob(target))
            logprob = (torch.log(self.mixing_coefficient * prob1 + (1-self.mixing_coefficient) * prob2)).sum()

        return logprob


class SampleDistribution(Distribution):
    """
    Collection of Gaussian predictions obtained by sampling
    """
    def __init__(self, predictions, var_noise):
        self.predictions = predictions
        self.var_noise = var_noise
        self.mean = self.predictions.mean()
        self.variance = self.predictions.var()


    def logprob(self, target):
        n_samples_testing = len(self.predictions)

        log_factor = -0.5 * np.log(2 * math.pi * self.var_noise) - (target - np.array(self.predictions))**2 / (2* self.var_noise)
        loglike = np.sum(logsumexp(log_factor - np.log(n_samples_testing)))

        return loglike

class BinarySampleDistribution(Distribution):
    """
    Collection of Bernoulli predictions obtained by sampling
    """
    def __init__(self, predictions):
        self.predictions = predictions
        self.mean = self.predictions.mean()
        self.point_estimate = round(self.mean)
        self.distributions = [Bernoulli(p) for p in predictions]

    def logprob(self, target):
        n_samples_testing = len(self.predictions)
        loglike = logsumexp(\
                  np.array([distr.logprob(target) for distr in self.distributions])\
                  - math.log(n_samples_testing))

        return loglike

class Bernoulli(Distribution):
    """ Bernoulli distribution """
    def __init__(self, probability):
        """
        Class constructor, sets parameters
        Args:
            probability: float, probability of observing 1
        Raises:
            ValueError: probability cannot be larger than 1
            ValueError: probability cannot be smaller than 0
        """
        if probability > 1:
            raise ValueError('Probability cannot be larger than 1')
        elif probability < 0:
            raise ValueError('Probability cannot be smaller than 0')
        elif not (isinstance(probability, float) or isinstance(probability, np.float32)):
            raise TypeError("Probability should be a float")

        self.mean = probability
        self.variance = probability * (1 - probability)

        if probability > 0.5:
            self.point_estimate = 1

    def logprob(self, target):
        """
        Computes the values of the predictive log likelihood at the target value
        Args:
            target: float, point to evaluate the logprob
        Returns:
            float, the log likelihood
        """
        if not (isinstance(target, np.integer) or isinstance(target, int)):
            raise TypeError("The given target should be an integer!")

        if target == 1:
            return np.log(self.mean)
        elif target == 0:
            return np.log(1 - self.mean)
        else:
            return - np.inf


class Gamma(Distribution):
    """ Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(shape, float) or not isinstance(rate, float):
            raise TypeError("Shape and rate should be floats!")

        if shape < 0 or rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.shape = shape
        self.rate = rate
        self.mean = self.shape / self.rate
        self.variance = self.shape / self.rate**2
        self.point_estimate = self.mean

    def sample(self, n_samples=1):
        # TODO: Double check this
        s = torch.distributions.Gamma(self.rate, self.shape).sample(sample_shape=(n_samples, *self.rate.size()))

        return s

    def update(self, shape, rate):
        """
        Updates mean and variance automatically when a and b get updated
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(shape, float) or not isinstance(rate, float):
            raise TypeError("Shape and rate should be floats!")

        if shape < 0 or rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.shape = shape
        self.rate = rate
        self.mean = shape / rate
        self.variance = shape / rate ** 2


class InvGaussian(Distribution):
    """
    Inverse Gaussian distribution
    """
    def __init__(self, mu, sigma):
        self.mean = mu
        self.std_dev = sigma
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        return 1 / (self.mean + self.std_dev * epsilon)

    def update(self, mean, sigma):
        """
        Updates mean and variance automatically
        Args:
            mean: float, mean of the inverse Gaussian
            sigma: standard deviation of the inverse Gaussian
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(mean, float) or not isinstance(sigma, float):
            raise TypeError("Mean and SD should be floats!")

        if mean < 0 or sigma < 0:
            raise ValueError("Shape and rate must be positive!")

        self.mean = mean
        self.std_dev = sigma


class GeneralizedInvGaussian(Distribution):
    """
    Inverse Gaussian distribution
    """
    def __init__(self, chi, rho, lamb):
        self.chi = chi
        self.rho = rho
        self.lamb = lamb

    def sample(self, n_samples=1):

        s = gigrnd(self.chi, self.rho, self.lamb)
        return s

    def update(self, chi, pho, lamb):
        """
        Updates mean and variance automatically
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        self.chi = chi
        self.pho = pho
        self.lamb = lamb


class InverseGamma(Distribution):
    """ Inverse Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters of the distribution.
        Args:
            shape: torch tensor of floats, shape parameters of the distribution
            rate: torch tensor of floats, rate parameters of the distribution
        """
        self.shape = shape
        self.rate = rate

    def exp_inverse(self):
        """
        Calculates the expectation E[1/x], where x follows
        the inverse gamma distribution
        """
        return self.shape / self.rate

    def exp_log(self):
        """
        Calculates the expectation E[log(x)], where x follows
        the inverse gamma distribution
        """
        exp_log = torch.log(self.rate) - torch.digamma(self.shape)
        return exp_log

    def entropy(self):
        """
        Calculates the entropy of the inverse gamma distribution
        """
        entropy =  self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                     - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def logprob(self, target):
        """
        Computes the value of the predictive log likelihood at the target value
        Args:
            target: Torch tensor of floats, point(s) to evaluate the logprob
        Returns:
            loglike: float, the log likelihood
        """
        part1 = (self.rate**self.shape) / gamma(self.shape)
        part2 = target**(-self.shape - 1)
        part3 = torch.exp(-self.rate / target)

        return torch.log(part1 * part2 * part3)

    def update(self, shape, rate):
        """
        Updates shape and rate of the distribution
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        """
        self.shape = shape
        self.rate = rate

class PredictiveDistribution:
    def __init__(self, distributions):
        """
        Class constructor, sets parameters
        Args:
           distributions: array of distributions
        """
        self.distributions = distributions

    def get_all_means(self):
        """
        extracts mean values from distributions
        Returns:
            array, means of distributions
        """
        means = [distr.mean for distr in self.distributions]

        return np.array(means)

    def get_all_variances(self):
        """
        extracts variances from distributions
        Returns:
            array, variances of distributions
        """
        variances = [distr.variance for distr in self.distributions]

        return np.array(variances)

    def get_all_point_estimates(self):
        """
        extracts point estimates from distributions
        Returns:
            array, point estimates of distributions
        """
        point_estimates = [distr.point_estimate for distr in self.distributions]

        return np.array(point_estimates)

    def get_all_predictions(self):
        """
        extracts predictions from distributions
        Returns:
            array, predictions of distributions
        """
        predictions = [distr.predictions for distr in self.distributions]

        return np.array(predictions)


class Exponential(Distribution):
    """ Exponential distribution """
    def __init__(self, rate):
        """
        Class constructor, sets parameters
        Args:
            rate: torch tensor of floats, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.rate = rate
        self.mean = 1 / self.rate
        self.variance = 1 / self.rate**2
        self.point_estimate = self.mean

    def sample(self, n_samples=1):
        s = torch.distributions.Exponential(self.rate).sample(sample_shape=(n_samples, *self.rate.size()))
        return s

    def update(self,  rate):
        """
        Updates mean and variance automatically when a and b get updated
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(rate, float):
            raise TypeError("Shape and rate should be floats!")

        if rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.rate = rate
        self.mean = 1 / rate
        self.variance = 1 / rate ** 2