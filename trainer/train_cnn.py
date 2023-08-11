"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
from torch.nn import functional as F

from .trainer import Trainer
import utils
from data import load_data

from parse import (
    parse_loss,
    parse_optimizer,
    parse_frequentist_model
)

import torchvision.transforms as transforms
import torchvision
import torch


class CNNTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_frequentist_model(self.config_model).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = F.nll_loss(log_outputs, label)
        loss.backward()

        self.optimzer.step()

        return loss.item(), log_outputs, label

    def valid_one_step(self, data, label):

        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)

        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = F.nll_loss(log_outputs, label)

        return loss.item(), log_outputs, label

    def validate(self):
        valid_acc_list = []
        probs = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            res, log_outputs, label = self.valid_one_step(data, label)

            probs.append(log_outputs)
            labels.append(label)

        probs = torch.cat(probs)
        labels = torch.cat(labels)
        test_metrics = utils.metrics(probs, labels, prefix="te")

        return test_metrics

    def train(self) -> None:
        print(f"Start training Frequentist CNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, log_outputs, label = self.train_one_step(data, label)

                training_loss_list.append(res)

                probs.append(log_outputs)
                labels.append(label)

            probs = torch.cat(probs)
            labels = torch.cat(labels)
            train_metrics = utils.metrics(probs, labels)
            test_metrics = self.validate()

            loss_dict = {
                "loss_tot": np.mean(training_loss_list),
            }

            training_range.set_description(
                'Epoch: {} \tTr Loss: {:.4f} \tTr Acc: {:.4f} \tVal Acc: {:.4f}'.format(
                    epoch, loss_dict["loss_tot"], train_metrics["tr_accuracy"],
                    test_metrics["te_accuracy"]))

            self.logging(epoch, loss_dict, train_metrics, test_metrics)

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
