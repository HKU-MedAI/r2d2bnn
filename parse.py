from torch import optim, nn
import torch.nn.functional as F

from models import (
    BBB3Conv3FC,
    BBBLeNet,
    BBBAlexNet2,
    BBBResNet,
    BBBConvNet,
    BBBHorseshoeLeNet,
    ResNet
)
from models.frequentists import LeNet, EfficientNetB4, AlexNet
from torchvision.models import resnet18, resnet50

from losses import ELBO, SSIMLoss


def parse_optimizer(config_optim, params):
    opt_method = config_optim["opt_method"].lower()
    alpha = config_optim["lr"]
    weight_decay = config_optim["weight_decay"]
    if opt_method == "adagrad":
        optimizer = optim.Adagrad(
            # model.parameters(),
            params,
            lr=alpha,
            lr_decay=weight_decay,
            weight_decay=weight_decay,
        )
    elif opt_method == "adadelta":
        optimizer = optim.Adadelta(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    elif opt_method == "adam":
        optimizer = optim.Adam(
            # model.parameters(),
            params,
            lr=alpha,
            # weight_decay=weight_decay,
        )
    else:
        optimizer = optim.SGD(
            # model.parameters(),
            params,
            lr=alpha,
            weight_decay=weight_decay,
        )
    return optimizer


def parse_loss(config_train):
    loss_name = config_train["loss"]

    if loss_name == "BCE":
        return nn.BCELoss()
    elif loss_name == "CE":
        return nn.CrossEntropyLoss()
    elif loss_name == "NLL":
        return nn.NLLLoss()
    elif loss_name == "ELBO":
        train_size = config_train["train_size"]
        return ELBO(train_size)
    elif loss_name == "SSIM":
        return SSIMLoss()
    elif loss_name == "cosine":
        return nn.CosineSimilarity(dim=-1)
    else:
        raise NotImplementedError("This Loss is not implemented")


def parse_bayesian_model(config_train, classifier: str = None):
    # Read input and output dimension
    in_dim = config_train["in_channels"]
    out_dim = config_train["out_channels"]

    model_name = config_train["model_name"]

    if model_name in ["BCNN", "BLeNet", "BAlexNet2"]:
        priors = {
            'prior_mu': config_train["prior_mu"],
            'prior_sigma': config_train["prior_sigma"],
            'posterior_mu_initial': config_train["posterior_mu_initial"],
            'posterior_rho_initial': config_train["posterior_rho_initial"],
        }

    if model_name == "BCNN":
        return BBB3Conv3FC(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    elif model_name == "BLeNet":
        return BBBLeNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    elif model_name == "HorseshoeLeNet":
        parameters = {
            "horseshoe_scale": config_train["horseshoe_scale"],
            "global_cauchy_scale": config_train["global_cauchy_scale"],
            "weight_cauchy_scale": config_train["weight_cauchy_scale"],
            "beta_rho_scale": config_train["beta_rho_scale"],
            "log_tau_mean": config_train["log_tau_mean"],
            "log_tau_rho_scale": config_train["log_tau_rho_scale"],
            "bias_rho_scale": config_train["bias_rho_scale"],
            "log_v_mean": config_train["log_v_mean"],
            "log_v_rho_scale": config_train["log_v_rho_scale"]
        }
        return BBBHorseshoeLeNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=parameters
        )
    elif model_name == "BAlexNet2":
        model = BBBAlexNet2(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
        return model
    elif model_name == "BConvNet":
        return BBBConvNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    elif model_name == "BResNet":
        return BBBResNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    else:
        raise NotImplementedError("This Model is not implemented")


def parse_frequentist_model(config_freq, image_size=32):
    # Read input and output dimension
    in_dim = config_freq["in_channels"]
    out_dim = config_freq["out_channels"]
    model_name = config_freq["model_name"]

    if model_name == "EfficientNet":
        return EfficientNetB4(
            inputs=in_dim,
            outputs=out_dim
        )
    elif model_name == "AlexNet":
        return AlexNet(
            inputs=in_dim,
            outputs=out_dim
        )
    elif model_name == "LeNet":
        return LeNet(
            outputs=out_dim,
            inputs=in_dim,
            image_size=image_size
        )
    elif model_name == "ResNet":
        return ResNet(
            outputs=out_dim,
            inputs=in_dim
        )
    else:
        raise NotImplementedError("This Loss is not implemented")
