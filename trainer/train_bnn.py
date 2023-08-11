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


class BNNTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size, self.config_data["image_size"])

        self.model = parse_bayesian_model(self.config_model).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label, beta):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)

        pred = self.model(data)
        kl_loss = self.model.kl_loss()

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)
        # log_outputs = F.log_softmax(pred, dim=1)

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)
        loss.backward()

        self.optimzer.step()

        return loss.item(), kl_loss.item(), nll_loss.item(), log_outputs

    def valid_one_step(self, data, label, beta):

        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)

        with torch.no_grad():
            pred = self.model(data)
            kl_loss = self.model.kl_loss()

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)

        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)

        return loss.item(), kl_loss.item(), nll_loss.item(), log_outputs

    def validate(self, epoch):
        valid_loss_list = []
        valid_kl_list = []
        valid_nll_list = []
        probs = []
        preds = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            # beta = utils.get_beta(i - 1, len(self.valid_loader), "Standard", epoch, self.n_epoch)
            beta = self.beta
            res, kl, nll, log_outputs = self.valid_one_step(data, label, beta)

            probs.append(log_outputs)
            preds.append(log_outputs)
            labels.append(label)

            valid_loss_list.append(res)
            valid_kl_list.append(kl)
            valid_nll_list.append(nll)

        probs = torch.cat(probs)
        labels = torch.cat(labels)

        test_metrics = utils.metrics(probs, labels, prefix="te")

        return test_metrics

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            probs = []
            labels = []

            beta = self.beta

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, kl, nll, log_outputs = self.train_one_step(data, label, beta)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)

                probs.append(log_outputs)
                labels.append(label)

            probs = torch.cat(probs)
            labels = torch.cat(labels)
            train_metrics = utils.metrics(probs, labels)

            test_metrics = self.validate(epoch)

            loss_dict = {
                "loss_tot": np.mean(training_loss_list),
                "loss_kl": np.mean(kl_list),
                "loss_ce": np.mean(nll_list)
            }

            self.logging(epoch, loss_dict, train_metrics, test_metrics)

            training_range.set_description(
                'Epoch: {} \tTr Loss: {:.4f} \tTr Acc: {:.4f} \tVal Acc: {:.4f} \tTr Kl Div: {:.4f} \tTr NLL: {:.4f}'.format(
                    epoch, loss_dict["loss_tot"], train_metrics["tr_accuracy"],
                    test_metrics["te_accuracy"], loss_dict["loss_kl"], loss_dict["loss_ce"]))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                }
                epoch_stats.update(loss_dict)
                epoch_stats.update(train_metrics)
                epoch_stats.update(test_metrics)

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
