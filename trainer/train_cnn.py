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

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size)

        self.model = parse_frequentist_model(self.config_train).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = F.nll_loss(log_outputs, label)
        loss.backward()

        self.optimzer.step()

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), acc, log_outputs, label

    def valid_one_step(self, data, label):

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)

        log_outputs = utils.logmeanexp(outputs, dim=2)

        loss = F.nll_loss(log_outputs, label)

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), acc, log_outputs, label

    def validate(self):
        valid_loss_list = []
        valid_acc_list = []
        probs = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))
            res, acc, log_outputs, label = self.valid_one_step(data, label)

            probs.append(log_outputs.softmax(1).detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            valid_loss_list.append(res)
            valid_acc_list.append(acc)

        valid_loss, valid_acc = np.mean(valid_loss_list), np.mean(valid_acc_list)
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        precision, recall, f1, aucroc = utils.metrics(probs, labels, average="weighted")

        return valid_loss, valid_acc, precision, recall, f1, aucroc

    def train(self) -> None:
        print(f"Start training Frequentist CNN...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            acc_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, acc, log_outputs, label = self.train_one_step(data, label)

                training_loss_list.append(res)
                acc_list.append(acc)

                probs.append(log_outputs.softmax(1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            train_loss, train_acc = np.mean(training_loss_list), np.mean(acc_list)
            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average="weighted")
            valid_loss, valid_acc, val_precision, val_recall, val_f1, val_aucroc = self.validate()

            training_range.set_description('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Train F1": train_f1,
                    "Train AUC": train_aucroc,
                    "Validation Loss": valid_loss,
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
