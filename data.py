import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms


def load_data(config_data, batch_size=32):
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
            transforms.Resize((32, 32)),
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
    elif config_data["name"] == "simulated":
        n = config_data["n"]
        trainset, testset  = simulate(n)

        # Dataloaders
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        raise NotImplementedError()

    return train_dataloader, valid_loader


def simulate(n):
    dataset = SimulatedDataset(n)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    return trainset, testset


class SimulatedDataset(Dataset):
    def __init__(self, n, mean=0, sd=1):
        self.n = n
        x = np.random.normal(loc=mean, scale=sd, size=(4, n))
        self.covariates = x.T.astype('float32')
        self.y = (1 / 4 * np.sin(x[0]) + x[1] + x[2] ** 3 + 1 / x[3]).astype('float32')

    def __len__(self):
        return self.covariates.shape[0]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.covariates[idx], self.y[idx]