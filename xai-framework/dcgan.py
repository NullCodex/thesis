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
import torchvision.utils as vutils

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.dataset == 'mnist':
    mnist = MNIST(root=opt.dataset_dir,
        train=True,
        download=True,
        transform=Compose([ToTensor(), Resize(28), Normalize(mean=(0.5,), std=(0.5,))]))
else:
    mnist = FashionMNIST(root=opt.dataset_dir,
         train=True,
         download=True,
         transform=Compose([ToTensor(), Resize(28), Normalize(mean=(0.5,), std=(0.5,))]))


dataloader = torch.utils.data.DataLoader(mnist, batch_size=64,
                                         shuffle=True, num_workers=2)

ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nc=1

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


class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class CGenerator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(CGenerator, self).__init__()
        self.ngpu = ngpu

        self.label_emb = nn.Embedding(10, 10)

        self.process_label = nn.Linear(10, 1)

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz + 1, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            embedded = self.label_emb(labels)
            attr = self.process_label(embedded).view(-1, 1, 1, 1)
            gen_input = torch.cat((input, attr), 1)
            output = self.main(gen_input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class NoiseDiscriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(NoiseDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            GaussianNoise(),
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            GaussianNoise(),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class CDiscriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(CDiscriminator, self).__init__()
        self.ngpu = ngpu

        self.label_emb = nn.Embedding(10, 10)
        self.process_label = nn.Linear(10, 28 * 28)

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            embedded = self.label_emb(labels)
            attr = self.process_label(embedded).view(-1, 1, 28, 28)
            x = torch.cat([input, attr], 1)
            output = self.main(x)
        return output.view(-1, 1).squeeze(1)


chosen_model = opt.model

if chosen_model == 0:
    netG = Generator(ngpu).to(device)

    netD = Discriminator(ngpu).to(device)
elif chosen_model == 1:
    netG = Generator(ngpu).to(device)
    netD = NoiseDiscriminator(ngpu).to(device)
else:
    netG = CGenerator(ngpu).to(device)
    netD = CDiscriminator(ngpu).to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0

if chosen_model <= 1:
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                       % (epoch, opt.n_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu, opt.output_dir + '/real_samples.png' ,normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), opt.output_dir + '/fake_samples_epoch_%03d.png' % (epoch), normalize=True)        
        torch.save(netG.state_dict(), opt.output_dir + '/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), opt.output_dir + '/netD_epoch_%d.pth' % (epoch))

else:
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            _label = torch.full((batch_size,), real_label, device=device)
            labels = data[1].to(device)
            output = netD(real_cpu, labels)
            errD_real = criterion(output, _label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise, labels)
            _label.fill_(fake_label)
            output = netD(fake.detach(), labels)
            errD_fake = criterion(output, _label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            _label.fill_(real_label)
            output = netD(fake, labels)
            errG = criterion(output, _label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                       % (epoch, opt.n_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu, opt.output_dir + '/real_samples.png' ,normalize=True)
                fake = netG(fixed_noise, labels)
                vutils.save_image(fake.detach(), opt.output_dir + '/fake_samples_epoch_%03d.png' % (epoch), normalize=True)        
        torch.save(netG.state_dict(), opt.output_dir + '/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), opt.output_dir + '/netD_epoch_%d.pth' % (epoch))
