"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
from data import load_data
import utils

from parse import (
    parse_loss,
    parse_optimizer,
    parse_bayesian_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class BNNHorseshoeTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_bayesian_model(self.config_train)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label, beta):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        log_prior = self.model.log_prior()
        log_variational_posterior = self.model.log_variational_posterior()

        kl_loss = (log_variational_posterior - log_prior).cuda()

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)
        loss.backward()

        self.optimzer.step()

        self.model.analytic_update()

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), kl_loss.item(), nll_loss.item(), acc, log_outputs, label

    def valid_one_step(self, data, label, beta):

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        log_prior = self.model.log_prior()
        log_variational_posterior = self.model.log_variational_posterior()
        kl_loss = (log_variational_posterior - log_prior).cuda()

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), kl_loss.item(), nll_loss.item(), acc, log_outputs, label

    def validate(self, epoch):
        valid_loss_list = []
        valid_kl_list = []
        valid_nll_list = []
        valid_acc_list = []
        probs = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            beta = self.beta
            res, kl, nll, acc, log_outputs, label = self.valid_one_step(data, label, beta)

            probs.append(log_outputs.softmax(1).detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            valid_loss_list.append(res)
            valid_kl_list.append(kl)
            valid_nll_list.append(nll)
            valid_acc_list.append(acc)

        valid_loss, valid_acc, valid_kl, valid_nll = np.mean(valid_loss_list), np.mean(valid_acc_list), np.mean(valid_kl_list), np.mean(valid_nll_list)
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        precision, recall, f1, aucroc = utils.metrics(probs, labels, average="weighted")

        return valid_loss, valid_acc, valid_kl, valid_nll, precision, recall, f1, aucroc

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            acc_list = []
            probs = []
            labels = []

            beta = self.beta

            for i, (data, label) in enumerate(self.dataloader):
                label = label.to(self.device)

                res, kl, nll, acc, log_outputs, label = self.train_one_step(data, label, beta)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)
                acc_list.append(acc)

                probs.append(log_outputs.softmax(1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            train_loss, train_acc, train_kl, train_nll = np.mean(training_loss_list), np.mean(acc_list), np.mean(
                kl_list), np.mean(nll_list)

            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average="weighted")

            valid_loss, valid_acc, valid_kl, valid_nll, val_precision, val_recall, val_f1, val_aucroc = self.validate(epoch)

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
                    "Train F1": train_f1,
                    "Train AUC": train_aucroc,
                    "Validation Loss": valid_loss,
                    "Validation KL Loss": valid_kl,
                    "Validation Accuracy": valid_acc,
                    "Validation F1": val_f1,
                    "Validation AUC": val_aucroc
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
