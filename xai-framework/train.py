import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch import LongTensor
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchvision.datasets import MNIST, FashionMNIST

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=int)

opt = parser.parse_args()
print(opt)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.dataset == 'mnist':
    if opt.model <= 2:
        mnist = MNIST(root=opt.dataset_dir,
                             train=True,
                             download=True,
                             transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    else:
        mnist = MNIST(root=opt.dataset_dir,
        train=True,
        download=True,
        transform=Compose([ToTensor(), Resize(32), Normalize(mean=(0.5,), std=(0.5,))]))
else:
    if opt.model <= 2:
        mnist = FashionMNIST(root=opt.dataset_dir,
                             train=True,
                             download=True,
                             transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
    else:
        mnist = FashionMNIST(root=opt.dataset_dir,
                             train=True,
                             download=True,
                             transform=Compose([ToTensor(), Resize(32), Normalize(mean=(0.5,), std=(0.5,))]))


#To ensure that the data is loaded in batches during training we are using a DataLoader.
batch_size = 100
data_loader = DataLoader(mnist, batch_size, shuffle=True)

image_size = 784
hidden_size = 256


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.main(X)

class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, X):
        return self.main(X)

class GaussianNoise(nn.Module):                         # Try noise just for real or just for fake images.
    def __init__(self, std=0.05, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

class DiscriminatorN(nn.Module):
    def __init__(self, image_size, hidden_size, std=0.1, std_decay_rate=0):
        super().__init__()
        self.std = std
        self.std_decay_rate = std_decay_rate

        self.main = nn.Sequential(
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            GaussianNoise(self.std, self.std_decay_rate),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.main(X)

class CGenerator(nn.Module):
    def __init__(self, latent_size, hidden_size):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.main = nn.Sequential(
            nn.Linear(latent_size + 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, X, labels):
        gen_input = torch.cat((X, self.label_emb(labels)), -1)
        return self.main(gen_input)

class CDiscriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super().__init__()
        self.label_embedding = nn.Embedding(10, 10)

        self.main = nn.Sequential(
            nn.Linear(image_size + 10, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X, labels):
        d_in = torch.cat((X, self.label_embedding(labels)), -1)
        return self.main(d_in)

class DCGenerator(nn.Module):
    def __init__(self, nc=1, nz=100, ngf=32):
        super(DCGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output

class DCDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=32):
        super(DCDiscriminator, self).__init__()
        self.network = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1)

model_type = opt.model

if model_type == 0:
    netD = Discriminator(image_size, hidden_size).to(device)
    netG = Generator(opt.latent_dim, hidden_size).to(device)
elif model_type == 1:
    netD = DiscriminatorN(image_size, hidden_size).to(device)
    netG = Generator(opt.latent_dim, hidden_size).to(device)
elif model_type == 2:
    netD = CDiscriminator(image_size, hidden_size).to(device)
    netG = CGenerator(opt.latent_dim, hidden_size).to(device)
else:
    netD = DCDiscriminator().to(device)
    netD.apply(weights_init)
    netG = DCGenerator().to(device)
    netG.apply(weights_init)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(netD.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(netG.parameters(), lr=0.0002)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

if model_type == 0 or model_type == 1:
    def train_discriminator(images):
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Loss for real images
        outputs = netD(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        fake_images = netG(z)
        outputs = netD(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        # Reset gradients
        reset_grad()
        # Compute gradients
        d_loss.backward()
        # Adjust the parameters using backprop
        d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator():
        # Generate fake images and calculate loss
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        fake_images = netG(z)
        labels = torch.ones(batch_size, 1).to(device)
        g_loss = criterion(netD(fake_images), labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        return g_loss, fake_images


    sample_vectors = torch.randn(batch_size, opt.latent_dim).to(device)

    def save_fake_images(index):
        fake_images = netG(sample_vectors)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        print('Saving', fake_fname)
        save_image(denorm(fake_images), os.path.join(opt.output_dir, fake_fname), nrow=10)

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(opt.n_epochs):
        for i, (images, _) in enumerate(data_loader):
            # Load a batch & transform to vectors
            images = images.reshape(batch_size, -1).to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, opt.n_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        # Sample and save images
        save_fake_images(epoch+1)

        torch.save(netG.state_dict(), opt.output_dir + '/generator-%s.pt' % str(epoch+1))
        torch.save(netD.state_dict(), opt.output_dir + '/discriminator-%s.pt' % str(epoch+1))
elif model_type == 2:
    def train_discriminator(images, labels):
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Loss for real images
        outputs = netD(images, labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        fake_images = netG(z, labels)
        outputs = netD(fake_images, labels)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        # Reset gradients
        reset_grad()
        # Compute gradients
        d_loss.backward()
        # Adjust the parameters using backprop
        d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator(labels):
        # Generate fake images and calculate loss
        z = torch.randn(batch_size, opt.latent_dim).to(device)
        fake_images = netG(z, labels)
        g_labels = torch.ones(batch_size, 1).to(device)
        g_loss = criterion(netD(fake_images, labels), g_labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        return g_loss, fake_images

    sample_vectors = torch.randn(batch_size, opt.latent_dim).to(device)

    def save_fake_images(index, generate_labels):
        fake_images = netG(sample_vectors, generate_labels)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        print('Saving', fake_fname)
        save_image(denorm(fake_images), os.path.join(opt.output_dir, fake_fname), nrow=10)

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(opt.n_epochs):
        for i, (images, labels) in enumerate(data_loader):
            # Load a batch & transform to vectors
            images = images.reshape(batch_size, -1).to(device)
            labels = labels.to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images, labels)
            g_loss, fake_images = train_generator(labels)

            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, opt.n_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        fixed_labels = LongTensor(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).to(device)
        fixed_labels = fixed_labels.repeat(10)
        save_fake_images(epoch+1, fixed_labels)

        torch.save(netG.state_dict(), opt.output_dir + '/generator-%s.pt' % str(epoch+1))
        torch.save(netD.state_dict(), opt.output_dir + '/discriminator-%s.pt' % str(epoch+1))

else:
    def train_discriminator(images):
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Loss for real images
        outputs = netD(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Loss for fake images
        z = torch.randn(batch_size, opt.latent_dim, 1, 1).to(device)
        fake_images = netG(z)
        outputs = netD(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Combine losses
        d_loss = d_loss_real + d_loss_fake
        # Reset gradients
        reset_grad()
        # Compute gradients
        d_loss.backward()
        # Adjust the parameters using backprop
        d_optimizer.step()

        return d_loss, real_score, fake_score

    def train_generator():
        # Generate fake images and calculate loss
        z = torch.randn(batch_size, opt.latent_dim, 1, 1).to(device)
        fake_images = netG(z)
        labels = torch.ones(batch_size, 1).to(device)
        g_loss = criterion(netD(fake_images), labels)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        return g_loss, fake_images


    sample_vectors = torch.randn(batch_size, opt.latent_dim, 1, 1).to(device)

    def save_fake_images(index):
        fake_images = netG(sample_vectors)
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
        print('Saving', fake_fname)
        save_image(denorm(fake_images), os.path.join(opt.output_dir, fake_fname), nrow=10)

    total_step = len(data_loader)
    d_losses, g_losses, real_scores, fake_scores = [], [], [], []

    for epoch in range(opt.n_epochs):
        for i, (images, _) in enumerate(data_loader):
            # Load a batch & transform to vectors
            images = images.to(device)

            # Train the discriminator and generator
            d_loss, real_score, fake_score = train_discriminator(images)
            g_loss, fake_images = train_generator()

            # Inspect the losses
            if (i+1) % 200 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                real_scores.append(real_score.mean().item())
                fake_scores.append(fake_score.mean().item())
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, opt.n_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        # Sample and save images
        save_fake_images(epoch+1)

        torch.save(netG.state_dict(), opt.output_dir + '/generator-%s.pt' % str(epoch+1))
        torch.save(netD.state_dict(), opt.output_dir + '/discriminator-%s.pt' % str(epoch+1))
