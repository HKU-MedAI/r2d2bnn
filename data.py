import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

from models import MLP


def load_data(config_data, batch_size=32, image_size=32):
    if config_data["name"] == "MNIST":
        transform_mnist = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        training_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
        valid_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
    elif config_data["name"] == "CIFAR10":
        transform_cifar = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=transform_cifar)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        valid_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
    elif config_data["name"] == "CIFAR100":
        transform_cifar = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        training_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_cifar)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)
        valid_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
    elif config_data["name"] == "Omniglot":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        training_data = torchvision.datasets.Omniglot(root='./data', background=True, download=True,
                                                     transform=transform)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.Omniglot(root='./data', background=False, download=True, transform=transform)
        valid_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
    elif config_data["name"] == "FashionMNIST":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        training_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                     transform=transform)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        valid_loader = DataLoader(testset, batch_size=batch_size, num_workers=4)
    elif config_data["name"] == "simulated":
        n = config_data["n"]
        scenario = config_data["scenario"]
        trainset, testset = simulate(n, scenario)

        # Dataloaders
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif config_data["name"] == "ImageNet":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
        testset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)
        # Dataloaders
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        raise NotImplementedError()

    return train_dataloader, valid_loader


def load_uncertainty_data(name, train, image_size, in_channel):
    if in_channel == 3:
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm = transforms.Normalize((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=in_channel),
            torchvision.transforms.ToTensor(),
            norm,
            torchvision.transforms.Resize(image_size)
        ])
    if name == "MNIST":
        d = torchvision.datasets.MNIST
    elif name == "FashionMNIST":
        d = torchvision.datasets.FashionMNIST
    elif name == "CIFAR10":
        d = torchvision.datasets.CIFAR10
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100
    elif name == "ImageNet":
        if train:
            return torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
        else:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(num_output_channels=in_channel),
                    torchvision.transforms.ToTensor(),
                    norm,
                    torchvision.transforms.Resize((image_size, image_size))
                ])
            return torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)
    elif name == "Omiglot":
        d = torchvision.datasets.Omniglot
        return d(root='./data', background=train, download=True, transform=transform)
    elif name == "SVHN":
        train = "train" if train else "test"
        d = torchvision.datasets.SVHN
        return d(root='./data', split=train, download=True, transform=transform)
    else:
        raise NotImplementedError

    return d(root='./data', train=train, download=True, transform=transform)

def simulate(n, scenario):
    dataset = SimulatedDataset(n, scenario)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    return trainset, testset


class SimulatedDataset(Dataset):
    def __init__(self, n, scenario, mean=0, sd=1):
        # Set random seed
        np.random.seed(30)
        self.n = n
        if scenario == 1:  # Polynomial case
            x = np.random.uniform(-5, 5, size=(1, n))
            eps = np.random.normal(loc=mean, scale=3, size=n)
            self.y = x[0] ** 3 + eps
        elif scenario == 2:  # Linear Regression
            x = np.random.uniform(-5, 5, size=(4, n))
            self.y = x[0] + 1 /4 * x[1] + 2 * x[2] ** 2 + x[3]
        elif scenario == 3:  # Non linear case
            x = np.random.uniform(-5, 5, size=(4, n))
            eps = np.random.normal(loc=mean, scale=3, size=n)
            self.y = x[0] * (x[1] ** 2 + 1) + x[2] * x[3] + eps
        elif scenario == 4:  # Sparse case
            p = 1000
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p * 9 // 10)
            beta_2 = np.random.uniform(0, 1, size=p // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
        elif scenario == 5:  # Dense case
            p = 1000
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p // 10)
            beta_2 = np.random.uniform(0, 1, size=p * 9 // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
        elif scenario == 6:  # Neural network case
            p = 1000
            net = MultipleLinear(1, p, 2)
            x = np.random.uniform(-5, 5, size=(p, n)).astype(np.float32)
            eps = np.random.normal(loc=mean, scale=2, size=n)

            self.y = net(torch.from_numpy(x.T))
            self.y = self.y.detach().numpy().squeeze() + eps
        else:
            raise NotImplementedError("This scenario is not implemented")

        self.y = self.y.astype('float32')

        assert x is not None

        self.covariates = x.T.astype('float32')

    def __len__(self):
        return self.covariates.shape[0]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.covariates[idx], self.y[idx]
