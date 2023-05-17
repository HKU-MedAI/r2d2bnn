from abc import ABC
from collections import OrderedDict

import numpy as np
import torch

from checkpoint import CheckpointManager
from matplotlib import pyplot as plt


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["data"]
        self.config_train = config['train']
        self.config_optim = config['optimizer']
        self.config_checkpoint = config['checkpoints']

        # Define checkpoints manager
        self.checkpoint_manager = CheckpointManager(self.config_checkpoint['path'])
        self.save_steps = self.config_checkpoint["save_checkpoint_freq"]

        # Load number of epochs
        self.n_epoch = self.config_train['num_epochs']
        self.starting_epoch = self.checkpoint_manager.version

        # Read batch size
        self.batch_size = self.config_train['batch_size']

        # Load device for training
        self.gpu_ids = config['gpu_ids']
        self.device = "cuda" if config['gpu_ids'] else "cpu"
        self.use_gpu = True if self.device == "cuda" else False

        self.model = None

    def train(self) -> None:
        raise NotImplementedError

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
