import argparse
import random

import torch
import yaml

from globals import *
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
)
from utils import ordered_yaml

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=612)

args = parser.parse_args()

opt_path = args.config
default_config_path = "HorseshoeMLP_Simulation.yml"

if opt_path == "":
    opt_path = CONFIG_DIR / default_config_path

# Set seed
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)

####
# Set types (train/eval)
####
mode = "train"


def main():
    # Load configurations
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    name = config["name"]
    for s in range(1, 6):
        for l in range(4):

            config["train"]["n_blocks"] = l
            config["data"]["scenario"] = s
            config["checkpoints"]["path"] = f"./checkpoints/Simulations/{name}/{name}MLP_L{l}_S{s}/"

            # Parse channels
            if s == 1:
                config["train"]["in_channels"] = 1
            elif s == 2 or s == 3:
                config["train"]["in_channels"] = 4
            else:
                config["train"]["in_channels"] = 200

            if mode == "train":
                if config["train_type"] == "bnn":
                    trainer = BNNTrainer(config)
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
                trainer.train()
            else:
                raise NotImplementedError("This mode is not implemented")

            with open(f"./checkpoints/Simulations/{name}/summary.txt", "a") as f:
                f.write(f"L: {l}, S: {s}"+ str(trainer.checkpoint_manager.stats) + "\n")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
