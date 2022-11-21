"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
from data import MVTecDataset
import utils

from parse import (
    parse_loss,
    parse_optimizer,
    parse_bayesian_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class BNNTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        if self.config_data["name"] == "MNIST":
            transform_mnist = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            training_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
            self.dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
            self.valid_loader = DataLoader(testset, batch_size=self.batch_size, num_workers=4)
        elif self.config_data["name"] == "CIFAR10":
            transform_cifar = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
            self.dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
            self.valid_loader = DataLoader(testset, batch_size=self.batch_size, num_workers=4)
        elif self.config_data["name"] == "CIFAR100":
            transform_cifar = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            training_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
            self.dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
            self.valid_loader = DataLoader(testset, batch_size=self.batch_size, num_workers=4)

        self.model = parse_bayesian_model(self.config_train).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.priors())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label, beta):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred, kl_loss = self.model(data)

        # outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        # log_outputs = utils.logmeanexp(outputs, dim=2)
        log_outputs = F.log_softmax(pred, dim=1)

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)
        loss.backward()

        self.optimzer.step()

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), kl_loss.item(), nll_loss.item(), acc

    def valid_one_step(self, data, label, beta):

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred, kl_loss = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)

        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), kl_loss.item(), nll_loss.item(), acc

    def validate(self, epoch):
        valid_loss_list = []
        valid_kl_list = []
        valid_nll_list = []
        valid_acc_list = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            beta = self.beta
            res, kl, nll, acc = self.valid_one_step(data, label, beta)

            valid_loss_list.append(res)
            valid_kl_list.append(kl)
            valid_nll_list.append(nll)
            valid_acc_list.append(acc)

        valid_loss, valid_acc, valid_kl, valid_nll = np.mean(valid_loss_list), np.mean(valid_acc_list), np.mean(valid_kl_list), np.mean(valid_nll_list)

        return valid_loss, valid_acc, valid_kl, valid_nll

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            acc_list = []

            beta = self.beta

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, kl, nll, acc = self.train_one_step(data, label, beta)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)
                acc_list.append(acc)

            train_loss, train_acc, train_kl, train_nll = np.mean(training_loss_list), np.mean(acc_list), np.mean(
                kl_list), np.mean(nll_list)

            valid_loss, valid_acc, valid_kl, valid_nll = self.validate(epoch)

            training_range.set_description('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \tTrain_kl_div: {:.4f} \tTrain_nll: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl, train_nll))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train NLL Loss": train_nll,
                    "Train KL Loss": train_kl,
                    "Train Accuracy": train_acc,
                    "Validation Loss": valid_loss,
                    "Validation KL Loss": valid_kl,
                    "Validation Accuracy": valid_acc,
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
