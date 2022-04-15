import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader

from dtw import SoftDTW
from data import RealData, extract_time, train_test_divide, SequenceData, DiscriminativeDataset
from models import Generator, Discriminator, DiscriminativeLSTM

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--seq_length", type=int, default=24)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--with_dtw", type=int, default=0)

opt = parser.parse_args()
print(opt)

average_discriminative = 0.0
for trial in range(5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_data_class = RealData(opt.dataset_dir)
    real_data = real_data_class.real_data_loading(opt.dataset, opt.seq_length)

    no, seq_len, dim = np.asarray(real_data).shape

    ori_time, ori_max_seq_len = extract_time(real_data)

    train_x, test_x, train_t, test_t = train_test_divide(real_data, ori_time)

    train_dataset = SequenceData(train_x)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size)

    netG = Generator(n_features=dim, n_layers=opt.num_layers).to(device)
    netD = Discriminator(n_features=dim, n_layers=opt.num_layers).to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    real_label = 1
    fake_label = 0

    sdtw = SoftDTW(use_cuda=True, gamma=0.1)

    for epoch in range(opt.n_epochs):
        for i, data in enumerate(train_loader, 0):
            # train with real
            netD.zero_grad()
            real = data.to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real.dtype, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn((batch_size, seq_len, opt.latent_dim), device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            if opt.with_dtw == 1:
                reshaped_fake = fake.reshape((fake.shape[0], seq_len, dim))
                dist_loss = sdtw(real, reshaped_fake)
                errG += dist_loss.mean()
            elif opt.with_dtw == 2:
                reshaped_fake = fake.reshape((fake.shape[0], seq_len, dim))
                dist_loss = sdtw(real, reshaped_fake)
                errG -= dist_loss.mean()
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.n_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    sample_size = 1000
    idx = np.random.permutation(len(train_x))[:sample_size]
    ori_data = np.asarray(train_x)[idx]

    netG.eval()
    netD.eval()

    noise = torch.randn((sample_size, seq_len, opt.latent_dim), device=device)
    generated_data = netG(noise).detach().cpu().numpy()

    no, seq_len, dim = ori_data.shape

    for i in range(sample_size):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(sample_size)] + ["blue" for i in range(sample_size)]

    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)

    # Plotting
    f, ax = plt.subplots(1)
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:sample_size], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1],
                c = colors[sample_size:], alpha = 0.2, label = "Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.savefig(opt.output_dir + '/' + '%s-%s-%s-pca.png' % (opt.dataset, opt.with_dtw, opt.seq_length))

    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)

    # TSNE analysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:sample_size,0], tsne_results[:sample_size,1],
                c = colors[:sample_size], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[sample_size:,0], tsne_results[sample_size:,1],
                c = colors[sample_size:], alpha = 0.2, label = "Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(opt.output_dir + '/' + '%s-%s-%s-tsne.png' % (opt.dataset, opt.with_dtw, opt.seq_length))

    dis_lstm = DiscriminativeLSTM(dim, dim//2, 2, 1).to(device)
    latent = torch.randn((train_x.shape[0] + test_x.shape[0], seq_len, opt.latent_dim), device=device)
    generated = netG(latent).detach().cpu().numpy()
    test_generated = generated[train_x.shape[0]:]
    generated = generated[:train_x.shape[0]]

    dis_dataset = DiscriminativeDataset(train_x, generated)
    dis_loader = DataLoader(dis_dataset, batch_size=128, shuffle=True)

    dis_criterion = torch.nn.BCELoss()
    dis_optimizer = torch.optim.Adam(dis_lstm.parameters())

    for epoch in range(50):
        for i, data in enumerate(dis_loader, 0):
            data_x = data[0].to(device)
            data_y = data[1].to(device)
            dis_optimizer.zero_grad()

            train_p = dis_lstm(data_x)
            dis_loss = dis_criterion(train_p, data_y)
            dis_loss.backward()

            dis_optimizer.step()

    dis_testset = DiscriminativeDataset(test_x, test_generated)
    dis_test_loader = DataLoader(dis_testset, batch_size=128)

    dis_lstm.eval()
    correct, total = 0, 0

    for i, data in enumerate(dis_test_loader, 0):
        data_x = data[0].to(device)
        data_y = data[1].to(device)

        preds = dis_lstm(data_x)
        preds = preds > 0.5
        total += data_x.shape[0]
        correct += (preds == data_y).sum().item()

    acc = correct / total
    average_discriminative += acc

f = open(opt.output_dir + "/discriminative_score.txt", "w+")
f.write(str(average_discriminative/5))
f.close()

