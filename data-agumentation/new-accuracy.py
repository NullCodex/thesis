import numpy as np
import torch
import torch.nn as nn
import argparse

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets as dsets

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--generated_dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--comparison', type=int)

config, _ = parser.parse_known_args()

print(config)

comparison = config.comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

cuda = True if torch.cuda.is_available() else False

dataset = config.dataset

num_epochs = 100 if dataset == 'fashion-mnist' else 200

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.classifier = models.resnet50(pretrained=False)
        self.linear = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.classifier(x)
        x = self.linear(x)
        return x

transformations = [
'False-translate-scale-horizontal_flip',
'rotate-translate-scale-horizontal_flip']

def get_transforms(transformStr):
    separated = transformStr.split('-')
    augmentations = []
    for aug in separated:
        if aug == 'rotate':
            augmentations.append(transforms.RandomAffine(degrees=10))
        elif aug == 'translate':
            augmentations.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))
        elif aug == 'scale':
            augmentations.append(transforms.RandomAffine(0, scale=(0.9, 1.1)))
        elif aug == 'horizontal_flip':
            augmentations.append(transforms.RandomHorizontalFlip(0.5))

    return augmentations


print('Normal images + transformations')

normalization = [0.5] if config.dataset == 'fashion-mnist' else [0.5, 0.5, 0.5]


if comparison == 0:
    # Normal images + transformations
    for transformation in transformations:
        print(transformation)
        base_transforms = [transforms.ToTensor()]
        if config.dataset == 'fashion-mnist':
            base_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        base_transforms.append(transforms.Normalize(normalization, normalization))
        additional_transforms = get_transforms(transformation)

        base_transforms.extend(additional_transforms)

        if dataset == 'fashion-mnist':
            training_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

            test_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=False,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )
        else:
            training_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='train',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

            test_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='test',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

        network = Net()
        network = network.to(device)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=128,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=True,
        )

        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        highest_accuracy = 0.0
        for epoch in range(num_epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data.to(device))
                loss = criterion(output, target.to(device))
                loss.backward()
                optimizer.step()

            network.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data.to(device))
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(device).data.view_as(pred)).sum()

            accuracy = 100. * correct / len(test_loader.dataset)
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
        print(highest_accuracy)




def get_generated_images(dataset, augmentations, transformation_list):
    root_path = '%s/%s' % (config.generated_dir, transformation)
    return dsets.ImageFolder(root=root_path, transform=transforms.Compose(transformation_list))


if comparison == 1:

    print('(Normal images + GAN images) + transformation')
    # (Normal images + GAN images) + transformations
    for transformation in transformations:
        print(transformation)
        base_transforms = [transforms.ToTensor()]
        if config.dataset == 'fashion-mnist':
            base_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        base_transforms.append(transforms.Normalize(normalization, normalization))

        additional_transforms = get_transforms(transformation)
        base_transforms.extend(additional_transforms)

        if dataset == 'fashion-mnist':
            training_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

            test_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=False,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )
        else:
            training_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='train',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )


            test_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='test',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

        generated_data = get_generated_images(dataset, 'False-False-False-False', base_transforms)

        combined = torch.utils.data.ConcatDataset([training_data, generated_data])

        network = Net()
        network = network.to(device)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=128,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=True,
        )

        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        highest_accuracy = 0.0
        for epoch in range(num_epochs):
            print(epoch)
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data.to(device))
                loss = criterion(output, target.to(device))
                loss.backward()
                optimizer.step()

            network.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data.to(device))
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(device).data.view_as(pred)).sum()

            accuracy = 100. * correct / len(test_loader.dataset)
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
        print(highest_accuracy)


if comparison == 2:
    print('Normal images + GAN transformations')
    for transformation in transformations:
        print(transformation)
        base_transforms = [transforms.ToTensor()]
        if config.dataset == 'fashion-mnist':
            base_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        base_transforms.append(transforms.Normalize(normalization, normalization))

        if dataset == 'fashion-mnist':
            training_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

            test_data = dsets.FashionMNIST(
                    config.dataset_dir + "/fashion-mnist",
                    train=False,
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )
        else:
            training_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='train',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )


            test_data = dsets.SVHN(
                    config.dataset_dir + "/svhn",
                    split='test',
                    download=True,
                    transform=transforms.Compose(base_transforms),
                )

        generated_data = get_generated_images(dataset, transformation, base_transforms)

        combined = torch.utils.data.ConcatDataset([training_data, generated_data])

        network = Net()
        network = network.to(device)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=128,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=128,
            shuffle=True,
        )

        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss()

        highest_accuracy = 0.0
        for epoch in range(num_epochs):
            for data, target in train_loader:
                optimizer.zero_grad()
                output = network(data.to(device))
                loss = criterion(output, target.to(device))
                loss.backward()
                optimizer.step()

            network.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = network(data.to(device))
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.to(device).data.view_as(pred)).sum()

            accuracy = 100. * correct / len(test_loader.dataset)
            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy
        print(highest_accuracy)

