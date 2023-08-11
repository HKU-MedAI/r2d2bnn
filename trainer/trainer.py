from abc import ABC
from collections import OrderedDict

import numpy as np
import torch

from checkpoint import CheckpointManager
from matplotlib import pyplot as plt

import wandb


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["data"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoints']
        self.config_logging = config["logging"]
        self.config_model = config["model"]

        # Define checkpoints manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Load number of epochs
        self.n_epoch = self.config_train['num_epochs']
        # self.starting_epoch = self.checkpoint_manager.version
        self.starting_epoch = 0

        # Read batch size
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

        self.model = None

    def train(self) -> None:
        raise NotImplementedError

    def initialize_logger(self, notes=""):
        name = "_".join(
            [
                self.config["name"],
                self.config_model["name"],
                self.config_data["name"],
            ]
        )
        tags = self.config["logging"]["tags"]
        wandb.init(name=name,
                   project='R2D2BNN',
                   notes=notes,
                   config=self.config,
                   tags=tags,
                   mode=self.config_logging["mode"]
                   )

    def visualize_conf_interval(self, pred, label, x):
        """
        Visualize confidence interval for regressors
        pred: (S, N)
        label (N) ground truth of the function
        """
        pth = self.checkpoint_manager.path / "conf_int.png"
        labels_path = self.checkpoint_manager.path / "labels.npy"
        pred_path = self.checkpoint_manager.path / "pred.npy"

        upper = np.max(pred, axis=0)
        lower = np.min(pred, axis=0)
        mean = np.mean(pred, axis=0)

        indices = np.argsort(x[:, 0])

        fig, ax = plt.subplots()
        ax.scatter(x[indices, 0], label[indices])
        ax.scatter(x[indices, 0], mean[indices])
        ax.fill_between(x[indices, 0], lower[indices], upper[indices], color='b', alpha=.5)

        plt.xlim([-5, 5])
        plt.ylim([-200, 200])

        # Save figure to checkpoint
        plt.savefig(pth, dpi=1200)

        # Save labels
        np.save(labels_path, label)
        np.save(pred_path, pred)

        plt.close()

    def logging(self, epoch, loss_dict, train_metrics, test_metrics):
        wandb.log({"epoch": epoch})
        wandb.log(loss_dict)
        wandb.log(train_metrics)
        wandb.log(test_metrics)
