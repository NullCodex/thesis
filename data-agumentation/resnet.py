import torch
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import argparse

import os
import time
import tqdm as tqdm
from torch.autograd import Variable

import torch.nn as nn
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
parser.add_argument("--dataset_dir", type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument("--dataset", type=str)
opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if opt.dataset == 'fashion-mnist':
    base_transforms = [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
else:
    base_transforms = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

transform_train = transforms.Compose(base_transforms)

transform_test = transforms.Compose(base_transforms)

if opt.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(
        root=opt.dataset_dir, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root=opt.dataset_dir, train=False, download=True, transform=transform_test)
elif opt.dataset == 'svhn':
    trainset = torchvision.datasets.SVHN(
        root=opt.dataset_dir, split='train', download=True, transform=transform_train)

    testset = torchvision.datasets.SVHN(
        root=opt.dataset_dir, split='test', download=True, transform=transform_test)
else:
    trainset = torchvision.datasets.FashionMNIST(
        root=opt.dataset_dir, train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.FashionMNIST(
        root=opt.dataset_dir, train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)

import torchvision.models as models

class ResClassifier(nn.Module):
    def __init__(self): #in_features=512
        super(ResClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.fc = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.resnet(x)
        out = self.fc(x)
        return out

netF = ResClassifier()

if torch.cuda.is_available():
    netF = netF.cuda()

opt_f = optim.SGD(netF.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()

train_accs = []
test_accs = []

def test_accuracy(data_iter, netF):
    """Evaluate testset accuracy of a model."""
    acc_sum,n = 0,0
    for (imgs, labels) in data_iter:
        # send data to the GPU if cuda is availabel
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        netF.eval()
        with torch.no_grad():
            labels = labels.long()
            acc_sum += torch.sum((torch.argmax(netF(imgs), dim=1) == labels)).float()
            n += labels.shape[0]
    return acc_sum.item()/n

for epoch in range(0, 100):
    n, start = 0, time.time()
    train_l_sum = torch.tensor([0.0], dtype=torch.float32)
    train_acc_sum = torch.tensor([0.0], dtype=torch.float32)
    for i, (imgs, labels) in tqdm.tqdm(enumerate(iter(trainloader))):
        netF.train()
        imgs = Variable(imgs)
        labels = Variable(labels)
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
            train_l_sum = train_l_sum.cuda()
            train_acc_sum = train_acc_sum.cuda()

        opt_f.zero_grad()

        label_hat = netF(imgs)

        # loss function
        loss= criterion(label_hat, labels)
        loss.backward()
        opt_f.step()

        # calcualte training error
        netF.eval()
        labels = labels.long()
        train_l_sum += loss.float()
        train_acc_sum += (torch.sum((torch.argmax(label_hat, dim=1) == labels))).float()
        n += labels.shape[0]

    test_acc = test_accuracy(iter(testloader), netF)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' \
          % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))
    train_accs.append((train_acc_sum/n).cpu().item())
    test_accs.append(test_acc)


train_accs = np.array(train_accs)
test_accs = np.array(test_accs)

with open(opt.output_dir + '/train_accuracy.npy', 'wb') as f:
    np.save(f, train_accs)


with open(opt.output_dir + '/test_accuracy.npy', 'wb') as f:
    np.save(f, test_accs)


