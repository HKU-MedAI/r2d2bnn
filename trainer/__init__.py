from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .trainer import Trainer
from .train_classification import ClassificationTrainer
from .train_cnn import CNNTrainer
from .train_bnn import BNNTrainer
from .train_MCdropout import CNNMCDropoutTrainer
from .train_bnn_linreg import BLinearRegTrainer
from .train_horseshoe_bnn import BNNHorseshoeTrainer
from .train_r2d2_bnn import R2D2BNNTrainer
from .r2d2_bnn_linreg import R2D2LinearRegTrainer
from .horseshoe_bnn_linreg import HorseshoeLinearRegTrainer
from .train_mcdropout_linreg import MCDLinearRegTrainer
from .uncertainty_bnn import BNNUncertaintyTrainer

__all__ = [
    'Trainer',
    'BNNTrainer',
]
