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


class ClassificationTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.initialize_logger()

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size, self.config_data["image_size"])

        self.model = parse_bayesian_model(self.config_model).to(self.device)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        # KL Annealing
        self.beta = self.config_train.get("beta") if self.config_train.get("beta") else 0

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        pred = self.model(data)
        kl_loss = self.model.kl_loss()

        log_outputs = self.reparameterize_output(data, pred)
        # log_outputs = F.log_softmax(pred, dim=1)
        nll_loss = F.nll_loss(log_outputs, label, reduction='mean')

        loss = nll_loss + self.beta * kl_loss
        loss.backward()

        self.optimzer.step()

        return loss.item(), kl_loss.item(), nll_loss.item(), log_outputs

    def validate(self):
        probs = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))

            with torch.no_grad():
                pred = self.model(data)
                log_outputs = self.reparameterize_output(data, pred)

            probs.append(log_outputs)
            labels.append(label)

        probs = torch.cat(probs)
        labels = torch.cat(labels)

        test_metrics = utils.metrics(probs, labels, prefix="te")

        return test_metrics

    def train(self) -> None:
        print(f"Start training {self.config['name']}...")

        training_range = tqdm(range(self.n_epoch))
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            probs = []
            labels = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))

                res, kl, nll, log_outputs = self.train_one_step(data, label)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)

                probs.append(log_outputs)
                labels.append(label)

            probs = torch.cat(probs)
            labels = torch.cat(labels)
            train_metrics = utils.metrics(probs, labels)

            test_metrics = self.validate()

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

    def reparameterize_output(self, data, pred):
        outputs = torch.zeros(data.shape[0], self.config_model["out_channels"], 1).to(self.device)
        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        return utils.logmeanexp(outputs, dim=2)