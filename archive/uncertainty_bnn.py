"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
import utils

from parse import (
    parse_loss,
    parse_optimizer,
    parse_bayesian_model
)

import torchvision.transforms as transforms
import torchvision
import torch

import scipy

import tensorflow_datasets as tfds

import bdlb
from bdlb.core.plotting import leaderboard
from bdlb.core.constants import DIABETIC_RETINOPATHY_DIAGNOSIS_URL_MEDIUM
from bdlb.core.constants import LEADERBOARD_DIR_URL


class BNNUncertaintyTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.dtask = bdlb.load(
            benchmark="diabetic_retinopathy_diagnosis",
            level="medium",
            batch_size=self.batch_size,
            data_dir="data",
            download_and_prepare=False,
        )

        tr, val, te = self.dtask.datasets
        self.ds_train = tfds.as_numpy(tr)
        self.ds_val, self.ds_test = val, te

        self.n_samples = self.config_train["n_samples"]
        self.model = parse_bayesian_model(self.config_train, image_size=256)
        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
        # introduces other beta computations
        self.beta = self.config_train["beta"]

    def predict(self, x):
        """Monte Carlo dropout uncertainty estimator.

        Args:
          x: `numpy.ndarray`, datapoints from input space,
            with shape [B, H, W, 3], where B the batch size and
            H, W the input images height and width accordingly.

        Returns:
          mean: `numpy.ndarray`, predictive mean, with shape [B].
          uncertainty: `numpy.ndarray`, ncertainty in prediction,
            with shape [B].
        """

        x = torch.Tensor(x).to(self.device).permute(0, 3, 1, 2)

        # Get shapes of data
        self.batch_size, _, _, _ = x.shape

        # Monte Carlo samples from different dropout mask at test time
        scores = [self.model(x) for _ in range(self.n_samples)]
        probs = [s.softmax(1)[:, 0].detach().cpu().numpy() for s in scores]
        mc_samples = np.asarray(probs).reshape(-1, self.batch_size)

        # Bernoulli output distribution
        dist = scipy.stats.bernoulli(mc_samples.mean(axis=0))

        # Predictive mean calculation
        mean = dist.mean()

        # Use predictive entropy for uncertainty
        uncertainty = dist.entropy()

        return mean, uncertainty

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        if pred.dim() > 2:
            pred = pred.mean(0)

        kl_loss = self.model.kl_loss()
        ce_loss = F.cross_entropy(pred, label, reduction='mean')

        loss = ce_loss + kl_loss.item() * self.beta
        # loss = ce_loss
        loss.backward()

        self.optimzer.step()

        if hasattr(self.model, "analytic_update"):
            self.model.analytic_update()

        acc = utils.acc(pred.data, label)

        return loss.item(), kl_loss.item(), ce_loss.item(), acc, pred

    def valid_one_step(self, data, label, beta):

        outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)

        outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        log_outputs = utils.logmeanexp(outputs, dim=2)

        kl_loss = self.model.kl_loss()
        loss, nll_loss, kl_loss = self.loss_fcn(log_outputs, label, kl_loss, beta)

        acc = utils.acc(log_outputs.data, label)

        return loss.item(), kl_loss.item(), nll_loss.item(), acc, log_outputs

    def validate(self, epoch):

        results = self.dtask.evaluate(
            self.predict,
            dataset=self.ds_test.take(10),
            name="gaussian_bnn"
        )

        return results

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.n_epoch))
        train_loader = [ex for ex in self.ds_train]
        for epoch in training_range:
            training_loss_list = []
            kl_list = []
            nll_list = []
            acc_list = []
            probs = []
            labels = []

            for i, ex in enumerate(train_loader):

                data = torch.Tensor(ex[0]).to(self.device)
                data = data.permute(0, 3, 1, 2)
                label = torch.LongTensor(ex[1]).to(self.device)

                res, kl, nll, acc, log_outputs = self.train_one_step(data, label)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)
                acc_list.append(acc)

                probs.append(log_outputs.softmax(1).detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

            train_loss, train_acc, train_kl, train_nll = np.mean(training_loss_list), np.mean(acc_list), np.mean(
                kl_list), np.mean(nll_list)

            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            train_precision, train_recall, train_f1, train_aucroc = utils.metrics(probs, labels, average="binary")

            results = self.validate(epoch)
            valid_acc = results['accuracy']['mean'].mean()
            valid_auc = results['auc']['mean'].mean()
            valid_acc_sd = results['accuracy']['std'].mean()
            valid_auc_sd = results['auc']['std'].mean()

            training_range.set_description('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation AUC: {:.4f} \tValidation Accuracy: {:.4f} \tTrain_kl_div: {:.4f} \tTrain_nll: {:.4f}'.format(
                    epoch, train_loss, train_acc, valid_auc, valid_acc, train_kl, train_nll))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                    "Train Loss": train_loss,
                    "Train NLL Loss": train_nll,
                    "Train KL Loss": train_kl,
                    "Train Accuracy": train_acc,
                    "Train F1": train_f1,
                    "Train AUC": train_aucroc,
                    "Validation Accuracy": valid_acc,
                    "Validation AUC": valid_auc,
                    "Validation Acc SD": valid_acc_sd,
                    "Validation AUC SD": valid_auc_sd
                }

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()
