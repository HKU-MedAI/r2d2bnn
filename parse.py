from torch import optim, nn
import torch.nn.functional as F

from models import (
    LeNet,
    AlexNet,
    ResNet,
    CNN,
    SimpleCNN,
    MLP,
    VGG,
)
from torchvision.models import resnet18, resnet50

from losses import ELBO


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


def parse_bayesian_model(config_model, image_size=32):
    # Read input and output dimension
    in_dim = config_model["in_channels"]
    out_dim = config_model["out_channels"]

    model_name = config_model["name"]
    layer_type = config_model["layer_type"]

    if layer_type == "Freq":
        priors = None
    else:
        priors = config_model["prior"]

    if model_name == "BCNN":
        n_blocks = config_model["n_blocks"]
        return BBB3Conv3FC(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "MLP":
        n_blocks = config_model["n_blocks"]
        return MLP(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=layer_type,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "LeNet":
        return LeNet(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=layer_type,
            priors=priors,
            image_size=image_size
        )
    elif model_name == "HorseshoeCNN":
        n_blocks = config_model["n_blocks"]
        return BBBHorseshoeCNN(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "HorseshoeSimpleCNN":
        return BBBHorseshoeSimpleCNN(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    elif model_name == "HorseshoeMLP":
        n_blocks = config_model["n_blocks"]
        return HorseshoeMultipleLinear(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "R2D2CNN":
        n_blocks = config_model["n_blocks"]
        return R2D2CNN(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks
        )
    elif model_name == "R2D2LeNet":
        return BBBR2D2LeNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            image_size=image_size
        )
    elif model_name == "R2D2AlexNet":
        return BBBR2D2AlexNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            r2d2_type=r2d2_type
        )
    elif model_name == "R2D2SimpleCNN":
        return R2D2SimpleCNN(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
    elif model_name == "R2D2MLP":
        n_blocks = config_model["n_blocks"]
        return R2D2MultipleLinear(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            n_blocks=n_blocks,
            r2d2_type=r2d2_type
        )
    elif model_name == "BAlexNet":
        model = BBBAlexNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors,
            image_size=image_size
        )
        return model
    elif model_name == "BHorseshoeAlexNet":
        model = BBBHorseshoeAlexNet(
            outputs=out_dim,
            inputs=in_dim,
            priors=priors
        )
        return model
    elif model_name == "ResNet":
        return ResNet(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    elif model_name == "VGG":
        return VGG(
            outputs=out_dim,
            inputs=in_dim,
            layer_type=config_model["layer_type"],
            priors=priors
        )
    else:
        raise NotImplementedError("This Model is not implemented")


def parse_frequentist_model(config_freq, image_size=32):
    # Read input and output dimension
    in_dim = config_freq["in_channels"]
    out_dim = config_freq["out_channels"]
    model_name = config_freq["name"]

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
            inputs=in_dim,
        )
    elif model_name == "CNN":
        n_blocks = config_freq["n_blocks"]
        return CNN(
            outputs=out_dim,
            inputs=in_dim,
            n_blocks=n_blocks
        )
    elif model_name == "MLP":
        n_blocks = config_freq["n_blocks"]
        return MultipleLinear(
            outputs=out_dim,
            inputs=in_dim,
            n_blocks=n_blocks
        )
    else:
        raise NotImplementedError("This Loss is not implemented")
