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
default_config_path = "R2D2MLP_Simulation.yml"

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

def parse_trainer(config):

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
        return trainer
    else:
        raise NotImplementedError("This mode is not implemented")

def simulate_baselines(config):
    name = config["name"]
    for seed in range(5):
        st = hash(seed)
        random.seed(st)
        torch.manual_seed(st)
        for s in range(1, 7):
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
                    config["train"]["in_channels"] = 1000

                trainer = parse_trainer(config=config)
                trainer.train()

                with open(f"./checkpoints/Simulations/{name}/summary_seed{seed}.txt", "a") as f:
                    f.write(f"L: {l}, S: {s}" + str(trainer.checkpoint_manager.stats) + "\n")


def simulate_hyperparameters(config):
    name = config["name"]
    for b in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for s in range(1, 7):
            for l in range(4):

                config["train"]["n_blocks"] = l
                config["data"]["scenario"] = s
                config["checkpoints"]["path"] = f"./checkpoints/Simulations_b/{name}/{name}MLP_L{l}_S{s}/"
                # # config["train"]["prior_phi_prob"] = alpha
                # config["train"]["beta_rho_scale"] = [rho, 0.05]
                # config["train"]["bias_rho_scale"] = [rho, 0.05]
                config["train"]["weight_omega_shape"] = b

                # Parse channels
                if s == 1:
                    config["train"]["in_channels"] = 1
                elif s == 2 or s == 3:
                    config["train"]["in_channels"] = 4
                else:
                    config["train"]["in_channels"] = 1000

                trainer = parse_trainer(config=config)
                trainer.train()

                with open(f"./checkpoints/Simulations_alpha/{name}/summary_b{b}.txt", "a") as f:
                    f.write(f"L: {l}, S: {s}" + str(trainer.checkpoint_manager.stats) + "\n")

def main():

    # Load configurations
    with open(opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {opt_path}")

    simulate_hyperparameters(config)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
