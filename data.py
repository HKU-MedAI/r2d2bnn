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
        scenario = config_data["scenario"]
        trainset, testset = simulate(n, scenario)

        # Dataloaders
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    else:
        raise NotImplementedError()

    return train_dataloader, valid_loader


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
            p = 200
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p * 9 // 10)
            beta_2 = np.random.uniform(0, 1, size=p // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
        elif scenario == 5:  # Dense case
            p = 200
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p // 10)
            beta_2 = np.random.uniform(0, 1, size=p * 9 // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
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
