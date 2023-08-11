import argparse
import random

import torch

from trainer import (
    CNNTrainer,
    CNNMCDropoutTrainer,
    BNNTrainer,
    BLinearRegTrainer,
    BNNHorseshoeTrainer,
    R2D2BNNTrainer,
    R2D2LinearRegTrainer,
    HorseshoeLinearRegTrainer,
    MCDLinearRegTrainer,
    BNNUncertaintyTrainer,
    ClassificationTrainer
)
from utils import load_config

####
# Set types (train/eval)
####
mode = "train"


def parse_trainer(config):
    if mode == "train":
        if config["train_type"] == "bnn":
            trainer = BNNTrainer(config)
        elif config["train_type"] == "classification":
            trainer = ClassificationTrainer(config)
        elif config["train_type"] == "cnn":
            trainer = CNNTrainer(config)
        elif config["train_type"] == "cnn-mc":
            trainer = CNNMCDropoutTrainer(config)
        elif config["train_type"] == "bnn-horseshoe":
            trainer = BNNHorseshoeTrainer(config)
        elif config["train_type"] == "bnn-r2d2":
            trainer = R2D2BNNTrainer(config)
        elif config["train_type"] == "bnn-linreg":
            trainer = BLinearRegTrainer(config)
        elif config["train_type"] == "r2d2-linreg":
            trainer = R2D2LinearRegTrainer(config)
        elif config["train_type"] == "horseshoe-linreg":
            trainer = HorseshoeLinearRegTrainer(config)
        elif config["train_type"] == "mcd-linreg":
            trainer = MCDLinearRegTrainer(config)
        elif config["train_type"] == "bnn-uncertainty":
            trainer = BNNUncertaintyTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
    else:
        raise NotImplementedError("This mode is not implemented")

    return trainer


def benchmark_datasets(config):
    in_datasets = ["CIFAR10"]
    out_datasets = ["FashionMNIST", "OMNIGLOT", "SVHN"]

    for in_data in in_datasets:
        for out_data in out_datasets:
            config["train"]["in_channel"] = 3

            config["dataset"]["in"] = in_data
            config["dataset"]["ood"] = out_data
            config["checkpoints"]["path"] = f"./checkpoints/BLeNet_OOD_{in_data}_{out_data}"

            trainer = parse_trainer(config)

            trainer.train()


def main():
    TASK = "Classification"
    DATASET = "CIFAR10"

    config_name = "GaussLeNet.yml"
    config = load_config(config_name, config_dir=f"./configs/{TASK}/{DATASET}/")

    seed = 512
    random.seed(seed)
    torch.manual_seed(seed)

    trainer = parse_trainer(config)
    trainer.train()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
