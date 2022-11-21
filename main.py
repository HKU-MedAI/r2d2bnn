import argparse
import random

import torch
import yaml

from evaluators import (
    BPNEvaluator,
    BPNSSLEvaluator,
    BPNMdisEvaluator,
    FrequentistEvaluator
)
from globals import *
from trainer import (
    BNNTrainer,
    BNNHorseshoeTrainer,
    FrequentistTrainer,
)
from utils import ordered_yaml

parser = argparse.ArgumentParser()
parser.add_argument('-config', type=str, help='Path to option YMAL file.', default="")
parser.add_argument('-seed', type=int, help='random seed of the run', default=612)

args = parser.parse_args()

opt_path = args.config
default_config_path = "HorseshoeLeNet_CIFAR10.yml"

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

    if mode == "train":
        if config["train_type"] == "bnn":
            trainer = BNNTrainer(config)
        elif config["train_type"] == "bnn-horseshoe":
            trainer = BNNHorseshoeTrainer(config)
        elif config["train_type"] == "freq":
            trainer = FrequentistTrainer(config)
        else:
            raise NotImplementedError(f"Trainer of type {config['train_type']} is not implemented")
        trainer.train()
    elif mode == "eval":
        if config["eval_type"] == "bpn":
            evaluator = BPNEvaluator(config)
        elif config["eval_type"] == "bpn-ssl":
            evaluator = BPNSSLEvaluator(config)
        elif config["eval_type"] == "bpn-mdis":
            evaluator = BPNMdisEvaluator(config)
        elif config["eval_type"] == "freq":
            evaluator = FrequentistEvaluator(config)
        else:
            raise NotImplementedError(f"Evaluator of type {config['eval_type']} is not implemented")
        # evaluator.plot_all()
        evaluator.plot_tpr_fpr()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
