from abc import ABC
from collections import OrderedDict


from checkpoint import CheckpointManager


class Trainer(ABC):
    def __init__(self, config: OrderedDict) -> None:
        # Categorize configurations
        self.config = config
        self.config_data = config["dataset"]
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

    def train(self) -> None:
        raise NotImplementedError
