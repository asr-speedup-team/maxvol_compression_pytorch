import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np


class DummyDatasetCifar10:
    def __init__(self, batch_size=4, data_root='./data'):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size = batch_size
        self.trainset = torchvision.datasets.CIFAR10(root=data_root,
                                                     train=True,
                                                     download=True,
                                                     transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root=data_root,
                                                    train=False,
                                                    download=True,
                                                    transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      drop_last=True,
                                                      num_workers=2)
        self.classes = np.array(['plane', 'car', 'bird', 'cat',
                                 'deer', 'dog', 'frog', 'horse',
                                 'ship', 'truck'])


class DummyModelCifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv1d(3, 15, 100)
        self.pool = nn.MaxPool1d(4, 4)
        # (, 15, 232)
        self.conv2 = nn.Conv1d(15, 30, 50)
        # (, 30, 45)
        self.fc1 = nn.Linear(30 * 45, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, self.num_classes)

    def forward(self, x):
        x = torch.flatten(x, -2)  # flatten last 2 dimensions
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
