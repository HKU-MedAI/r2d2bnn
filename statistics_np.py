from math import sqrt

import numpy as np
from numpy import linalg

"""
ref: https://arxiv.org/pdf/1609.08725.pdf
"""


class AdaptableRHT:
    def __init__(self, lambdas, cov, n, p):
        self.lambdas = lambdas
        self.lambda_dict = {lamb: i for i, lamb in enumerate(lambdas)}
        self.n = n
        self.p = p
        self.gamma = p / n

        self.cov = cov

        self.reg_inv_covs = [
            self.regularized_inverse(-lamb, cov)
            for lamb in self.lambdas
        ]
        self.mz_list = [
            self.weighted_trace(self.reg_inv_covs[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]
        self.theta_1 = [
            self.get_theta_1(lamb, self.mz_list[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]
        self.theta_2 = [
            self.get_theta_2(lamb,
                             self.reg_inv_covs[self.lambda_dict[lamb]],
                             self.mz_list[self.lambda_dict[lamb]])
            for lamb in self.lambdas
        ]

    def regularized_inverse(self, z, cov):
        reg_cov = cov - z * np.eye(self.p)
        return linalg.inv(reg_cov)


    def weighted_trace(self, reg_inv_cov):
        m_z = np.trace(reg_inv_cov) / self.p
        return m_z


    def get_theta_1(self, lamb, m_z):
        a = 1 - lamb * m_z
        return a / (1 - self.gamma * a)

    def get_theta_2(self, lamb, reg_inv_cov, m_z):
        a = 1 - lamb * m_z
        reg_inv_sq = reg_inv_cov @ reg_inv_cov
        m_prime_z = np.trace(reg_inv_sq) / self.p
        th = a / (1 - self.gamma * a) ** 3 - lamb * (m_z - lamb * m_prime_z) / (1 - self.gamma * a) ** 4
        if th <= 0:
            th = -th + 10 ** -10
        return th

    def rht(self, lamb, mu_1, mu_2, n_1, n_2):
        reg_inv = self.reg_inv_covs[self.lambda_dict[lamb]]
        rht = n_1 * n_2 / (n_1 + n_2) * np.matmul(np.matmul((mu_1 - mu_2), reg_inv), (mu_1 - mu_2).T)
        return rht


    def adaptive_rht(self, lamb, mu_1, mu_2, n_1, n_2, p):
        t_stat = sqrt(p) * (self.rht(lamb, mu_1, mu_2, n_1, n_2)
                            / p - self.theta_1[self.lambda_dict[lamb]]) / sqrt(2 * self.theta_2[self.lambda_dict[lamb]])

        return t_stat


    def Q_function(self, lamb, priors):
        """
        Q function for tuning lambda from data
        :param lamb:
        :param cov:
        :param n:
        :param p:
        :param priors: prior weights in list of 3
        :return:
        """
        phi = np.trace(self.cov) / self.p

        rho_1 = self.mz_list[self.lambda_dict[lamb]]
        rho_2 = self.theta_1[self.lambda_dict[lamb]]
        th_1 = rho_2
        rho_3 = (1 + self.gamma * th_1) * (phi - lamb * rho_1)

        Q = [pri * rho / sqrt(self.gamma * self.theta_2[self.lambda_dict[lamb]]) for pri, rho in zip(priors, (rho_1, rho_2, rho_3))]
        Q = sum(Q).item()

        return Q
