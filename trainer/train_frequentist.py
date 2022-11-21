"""
Training of Frquentist replication of our method
Only one encoder for all patches
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
    parse_frequentist_model
)
from checkpoint import CheckpointManager

import torch


class FrequentistTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        # Load image size and window size
        self.image_size = self.config_data["image_size"]
        self.window_size = self.config_data["window_size"]

        # Number of rows and columns of the grid of patches
        self.n_rows = self.image_size // self.window_size
        self.n_cols = self.image_size // self.window_size

        # Load model and optimizer
        self.model = parse_frequentist_model(self.config_train).to(self.device)

        # Load pretrained models if any
        self.pretrained_checkpoint_path = self.config_checkpoint["init_path"]
        if self.pretrained_checkpoint_path:
            pretrained_checkpoint = CheckpointManager(self.pretrained_checkpoint_path)
            # One state dict for pretrained model for one window
            state_dict = pretrained_checkpoint.load_model()

            st_dic = self.model.state_dict()
            st_dict_keys = list(st_dic.keys())
            for l, (k, v) in enumerate(state_dict.items()):
                if 2 * l < len(st_dic) and st_dic[st_dict_keys[2 * l]].shape == v.shape:
                    st_dic[st_dict_keys[2 * l]] = v
            state_dict = st_dic

            self.model.load_state_dict(state_dict)

        self.optimizer = parse_optimizer(self.config_optim, self.model.priors())

        self.checkpoint_manager.n_models = 1

        data_dir = self.config_data["train_path"]
        training_data = MVTecDataset(data_dir, self.image_size, train=True)
        self.dataloader = DataLoader(training_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        test_data_dir = self.config_data["train_path"]
        testset = MVTecDataset(test_data_dir, self.image_size, valid=True)
        self.valid_loader = DataLoader(testset, batch_size=self.batch_size, num_workers=4, shuffle=True, drop_last=True)

        # Loss function for classification
        print(len(training_data))
        self.loss_fcn = parse_loss(self.config_train)

    def train_one_step(self, data, label):

        self.optimizer.zero_grad()

        pred = self.model(data)[0]

        # outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        # log_outputs = utils.logmeanexp(outputs, dim=2)
        log_outputs = F.log_softmax(pred, dim=1)

        loss = self.loss_fcn(log_outputs, label)
        loss.backward()

        self.optimizer.step()

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), acc

    def validate_one_step(self, data, label):

        pred = self.model(data)[0]
        log_outputs = F.log_softmax(pred, dim=1)

        loss = self.loss_fcn(log_outputs, label)

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), acc

    def validate(self):
        valid_loss_list = []
        valid_acc_list = []

        for i, (data, label) in enumerate(self.valid_loader):
            (data, label) = (data.to(self.device), label.to(self.device))

            input_data = [
                data[:, :, i * self.window_size:(i + 1) * self.window_size,
                j * self.window_size:(j + 1) * self.window_size]
                for i in range(self.n_rows)
                for j in range(self.n_cols)
            ]
            input_data = torch.cat(input_data, dim=0)
            label = label.repeat(input_data.shape[0] // self.batch_size)

            res, acc = self.validate_one_step(input_data, label)

            valid_loss_list.append(res)
            valid_acc_list.append(acc)

        valid_loss, valid_acc = np.mean(valid_loss_list), np.mean(valid_acc_list)
        return valid_loss, valid_acc

    def train(self) -> None:
        print(f"Start training Frequentist model...")

        for epoch in range(self.n_epoch):

            training_loss_list = []
            acc_list = []

            for i, (data, label) in enumerate(self.dataloader):
                (data, label) = (data.to(self.device), label.to(self.device))
                data_gl = F.interpolate(data, size=self.window_size)

                input_data = [
                    data[:, :, i * self.window_size:(i + 1) * self.window_size,
                    j * self.window_size:(j + 1) * self.window_size]
                    for i in range(self.n_rows)
                    for j in range(self.n_cols)
                ]
                input_data += [data_gl] * (len(input_data) // 2)

                label = label.repeat(len(input_data))
                input_data = torch.cat(input_data, dim=0)

                res, acc = self.train_one_step(input_data, label)

                training_loss_list.append(res)
                acc_list.append(acc)

            train_loss, train_acc = np.mean(training_loss_list), np.mean(acc_list)
            valid_loss, valid_acc = self.validate()

            print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} '.format(
                    epoch, train_loss, train_acc, valid_loss, valid_acc))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Validation Loss": valid_loss,
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
