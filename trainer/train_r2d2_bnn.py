"""
Trainer of BNN
"""
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .trainer import Trainer
import utils
from data import load_data

from parse import (
    parse_loss,
    parse_optimizer,
    parse_bayesian_model
)

import wandb

from matplotlib import pyplot as plt


class R2D2BNNTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        # Initialize logger
        self.initialize_logger()

        self.dataloader, self.valid_loader = load_data(self.config_data, self.batch_size, image_size=self.config_data["image_size"])

        self.model = parse_bayesian_model(self.config_model)
        # Load state dict if any
        # if self.checkpoint_manager.version > 0:
        #     sd = self.checkpoint_manager.load_model()
        #     self.model.load_state_dict(sd)

        self.optimzer = parse_optimizer(self.config_optim, self.model.parameters())

        self.loss_fcn = parse_loss(self.config_train)

        # Define beta for ELBO computations
        self.beta = self.config_train["beta"]

    def train_one_step(self, data, label):
        self.optimzer.zero_grad()

        # outputs = torch.zeros(data.shape[0], self.config_train["out_channels"], 1).to(self.device)

        pred = self.model(data)
        pred = pred.mean(0)  # Taking the mean over sample dimensions

        # outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        # pred = utils.logmeanexp(outputs, dim=2)
        # pred = F.normalize(pred, 1)  # Temporary solutions normalizing the graidents
        kl_loss = self.model.kl_loss()
        ce_loss = F.cross_entropy(pred, label, reduction='mean')

        loss = ce_loss + kl_loss.item() * self.beta
        # loss = ce_loss
        loss.backward()

        self.optimzer.step()

        return loss.item(), kl_loss.item(), ce_loss.item(), pred

    def train(self) -> None:
        print(f"Start training BNN...")

        training_range = tqdm(range(self.starting_epoch, self.n_epoch))
        for epoch in training_range:
            probs = []
            labels = []
            training_loss_list = []
            kl_list = []
            nll_list = []

            # if epoch % 5 == 0:
            #     self.visualize_map(epoch)

            for i, (data, label) in tqdm(enumerate(self.dataloader)):
                label = label.to(self.device)

                res, kl, nll, log_outputs = self.train_one_step(data, label)

                training_loss_list.append(res)
                kl_list.append(kl)
                nll_list.append(nll)

                if i % 10 == 0:
                    self.model.analytic_update()

                probs.append(log_outputs)
                labels.append(label)

            loss_dict = {
                "loss_tot": np.mean(training_loss_list),
                "loss_kl": np.mean(kl_list),
                "loss_ce": np.mean(nll_list)
            }
            probs = torch.cat(probs)
            labels = torch.cat(labels)
            train_metrics = utils.metrics(probs, labels)
            test_metrics = self.validate(epoch)

            self.logging(epoch, loss_dict, train_metrics, test_metrics)

            training_range.set_description(
                'Epoch: {} \tTr Loss: {:.4f} \tTr Acc: {:.4f} \tVal Acc: {:.4f} \tTr Kl Div: {:.4f} \tTr NLL: {:.4f}'.format(
                    epoch, loss_dict["loss_tot"], train_metrics["tr_accuracy"],
                    test_metrics["te_accuracy"], loss_dict["loss_kl"], loss_dict["loss_ce"]))

            # Update new checkpoints and remove old ones
            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                epoch_stats = {
                    "Epoch": epoch + 1,
                }
                epoch_stats.update(loss_dict)
                epoch_stats.update(train_metrics)
                epoch_stats.update(test_metrics)

                # State dict of the model including embeddings
                self.checkpoint_manager.write_new_version(
                    self.config,
                    self.model.state_dict(),
                    epoch_stats
                )

                # Remove previous checkpoints
                self.checkpoint_manager.remove_old_version()

    def validate(self, epoch):
        probs = []
        labels = []

        for i, (data, label) in enumerate(self.valid_loader):
            label = label.to(self.device)
            log_outputs = self.valid_one_step(data)

            probs.append(log_outputs)
            labels.append(label)

        probs = torch.cat(probs)
        labels = torch.cat(labels)
        val_metrics = utils.metrics(probs, labels, prefix="te")

        return val_metrics

    def valid_one_step(self, data):

        with torch.no_grad():
            pred = self.model(data)
            pred = pred.mean(0)

        # outputs[:, :, 0] = F.log_softmax(pred, dim=1)
        #
        # log_outputs = utils.logmeanexp(outputs, dim=2)
        pred = F.normalize(pred, 1)
        return pred

    def visualize_map(self, epoch):
        """
        Visualize the convolutional map of the features
        """

        fig, ax = plt.subplots(4, 5, figsize=(36, 10))

        pth = self.checkpoint_manager.path / "visualizations"
        if not pth.exists():
            pth.mkdir()

        # layer = self.model.convs[0]
        try:
            layer = self.model.conv1
        except AttributeError:
            return

        # Sample maps here
        beta = layer.beta_.sample(200)
        beta_sigma = layer.beta_.std_dev.detach()
        beta_eps = torch.empty(beta.size()).normal_(0, 1)
        beta_std = torch.sqrt(beta_sigma ** 2 * layer.omega * layer.phi * layer.psi / 2)

        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=beta.shape)
        weight = layer.beta_.mean + beta_std * epsilon
        weight = weight.detach().cpu().numpy().squeeze()
        weight = weight.mean(0)
        norms = np.linalg.norm(weight, axis=(1, 2))
        max_indices = torch.from_numpy(norms).topk(5).indices
        min_indices = (- torch.from_numpy(norms)).topk(5).indices

        for i in range(5):
            ax[0, i].imshow(weight[max_indices[i]], cmap='gray',  vmin=weight.min(), vmax=weight.max())
            ax[1, i].imshow(weight[min_indices[i]], cmap='gray', vmin=weight.min(), vmax=weight.max())

        d = self.dataloader.dataset[0][0].unsqueeze(0).cuda()
        mp = self.model.get_map(d).squeeze().detach().cpu().numpy()
        norms = np.linalg.norm(mp, axis=(1, 2))
        max_indices = torch.from_numpy(norms).topk(5).indices
        min_indices = (- torch.from_numpy(norms)).topk(5).indices

        # fig, ax = plt.subplots()
        for i in range(5):
            ax[2, i].imshow(mp[max_indices[i]], cmap='gray')
            ax[3, i].imshow(mp[min_indices[i]], cmap='gray')

        wandb.log(fig)
        plt.savefig(pth / f"MNIST_ep{epoch}.png")

        return

    def logging(self, epoch, loss_dict, train_metrics, test_metrics):
        wandb.log({"epoch": epoch})
        wandb.log(loss_dict)
        wandb.log(train_metrics)
        wandb.log(test_metrics)
