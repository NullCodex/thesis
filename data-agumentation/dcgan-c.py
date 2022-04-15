
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import argparse

parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

parser.add_argument('--epochs', type=int, default=500, help='The number of epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
parser.add_argument('--channels', type=int, default=1)

parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generator in WGAN model.')

parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--transforms', nargs="+", default=[])
parser.add_argument('--dataset', type=str, default='mnist')

config, _ = parser.parse_known_args()

print(config)

chosen_data = config.dataset

os.makedirs(config.dataset_dir + "/" + chosen_data, exist_ok=True)

base_transforms = [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]

chosen_transforms = config.transforms

ngf = 64

# Size of feature maps in discriminator
ndf = 64

if 'rotate' in chosen_transforms:
    base_transforms.append(transforms.RandomAffine(degrees=10))
elif 'translate' in chosen_transforms:
    base_transforms.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))
elif 'scale' in chosen_transforms:
    base_transforms.append(transforms.RandomAffine(0, scale=(0.9, 1.1)))
elif 'horizontal_flip' in chosen_transforms:
    base_transforms.append(transforms.RandomHorizontalFlip(0.5))

dataset_to_use = dset.SVHN(
    config.dataset_dir + "/fashion-mnist",
    split='train',
    download=True,
    transform=transforms.Compose(base_transforms),
    )

# Configure data loader
train_loader = torch.utils.data.DataLoader(
    dataset_to_use,
    batch_size=config.batch_size,
    shuffle=True,
)

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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.label_emb = nn.Embedding(10, 10)
        self.process_label = nn.Linear(10, 1)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(101, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        embedded = self.label_emb(labels)
        attr = self.process_label(embedded).view(-1, 1, 1, 1)
        gen_input = torch.cat((input, attr), 1)

        return self.main(gen_input)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.label_emb = nn.Embedding(10, 10)
        self.process_label = nn.Linear(10, 32 * 32)

        self.main = nn.Sequential(
            GaussianNoise(),
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            GaussianNoise(),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        embedded = self.label_emb(labels)
        attr = self.process_label(embedded).view(-1, 1, 32, 32)
        x = torch.cat([input, attr], 1)
        return self.main(x).view(-1)

device = torch.device("cuda:0")

netG = Generator()
netG.apply(weights_init_normal)
netG.to(device)
netD = Discriminator()
netD.apply(weights_init_normal)
netD.to(device)

criterion = nn.BCELoss()

# Adam optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

outf = config.output_dir

g_losses = []
real_d_losses = []
fake_d_losses = []
real_label = 1.0
fake_label = 0.0

fixed_noise = torch.randn(64, 100, 1, 1, device=device)

for epoch in range(config.epochs):
    for i, data in enumerate(train_loader, 0):
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
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
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
              % (epoch, config.epochs, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu, config.output_dir + '/real_samples.png' ,normalize=True)
            fake = netG(fixed_noise, labels)
            vutils.save_image(fake.detach(), config.output_dir + '/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
    torch.save(netG.state_dict(), config.output_dir + '/netG_epoch_%d.pth' % (epoch))
    torch.save(netD.state_dict(), config.output_dir + '/netD_epoch_%d.pth' % (epoch))


# for epoch in range(200):
#
#     if (epoch+1) == 11:
#         optimizerG.param_groups[0]['lr'] /= 10
#         optimizerD.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#
#     if (epoch+1) == 16:
#         optimizerG.param_groups[0]['lr'] /= 10
#         optimizerD.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#
#     average_real_d_loss = 0.0
#     average_fake_d_loss = 0.0
#     average_g_loss = 0.0
#     counter = 0
#
#     for i, data in enumerate(train_loader, 0):
#         ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         ###########################
#         # train with real
#         netD.zero_grad()
#         real_cpu = data[0].to(device)
#         real_labels = data[1].to(device)
#         batch_size = real_cpu.size(0)
#         label = torch.full((batch_size,), real_label,
#                            dtype=real_cpu.dtype, device=device)
#
#         real_labels = fill[real_labels]
#         output = netD(real_cpu, real_labels)
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()
#
#         # train with fake
#         noise = torch.randn(batch_size, 100, 1, 1, device=device)
#         fake_classes = torch.randint(0, 10, (batch_size,), device=device)
#         one_hot_classes = onehot[fake_classes]
#
#         fake = netG(noise, one_hot_classes)
#         label.fill_(fake_label)
#         fake_classes = fill[fake_classes]
#         output = netD(fake.detach(), fake_classes)
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         errD = errD_real + errD_fake
#         optimizerD.step()
#
#         ############################
#         # (2) Update G network: maximize log(D(G(z)))
#         ###########################
#         netG.zero_grad()
#         label.fill_(real_label)  # fake labels are real for generator cost
#         output = netD(fake, fake_classes)
#         errG = criterion(output, label)
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         optimizerG.step()
#
#         print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
#               % (epoch, 100, i, len(train_loader),
#                  errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
#         average_g_loss += errG.item()
#         average_fake_d_loss += D_G_z1
#         average_real_d_loss += D_G_z2
#
#         if i % 1000 == 0:
#             vutils.save_image(real_cpu,
#                               '%s/real_samples.png' % outf,
#                               normalize=True)
#             fake = netG(fixed_noise, fixed_classes)
#             vutils.save_image(fake.detach(),
#                               '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
#                               normalize=True)
#
#         counter += 1
#
#     g_losses.append(average_g_loss / counter)
#     real_d_losses.append(average_real_d_loss / counter)
#     fake_d_losses.append(average_fake_d_loss / counter)
#
#     # do checkpointing
#     torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
#     torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
#
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# plt.plot(real_d_losses, label='d_real_loss')
# plt.plot(fake_d_losses, label='d_fake_loss')
# plt.plot(g_losses, label='g_loss')
# plt.legend(loc='upper center')
# plt.savefig(config.output_dir + "/" + 'losses.png')

