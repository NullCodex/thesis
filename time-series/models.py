import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent=100, n_features=5, n_hidden=32, n_layers=1):
        super(Generator, self).__init__()
        self.z = latent
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.latent = nn.Linear(latent, 128)

        self.time = nn.LSTM(latent, n_hidden, n_layers, batch_first=True)

        self.generate = nn.Sequential(
            nn.Linear(n_hidden, n_features),
            nn.Sigmoid()
        )

    # z - Batch size x Length x n_features
    def forward(self, Z):
        T = np.full(Z.shape[0], Z.shape[1])
        Z_packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=Z,
            lengths=T,
            batch_first=True,
            enforce_sorted=False
        )

        H_packed, H_t = self.time(Z_packed)

        # Pad RNN output back to sequence length
        H_o, T = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=H_packed,
            batch_first=True,
            padding_value=-1.0,
            total_length=Z.shape[1]
        )

        return self.generate(H_o)


class Discriminator(nn.Module):
    def __init__(self, n_features=5, n_hidden=32, n_layers=1):
        super(Discriminator, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.time = nn.LSTM(n_features, n_hidden, n_layers, batch_first=True)

        self.generated = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid()
        )

        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.time(x, (h0, c0))
        out = self.generated(out[:, -1, :])
        return out.view(-1)

    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden)
        return [t.cuda() for t in (h0, c0)]


class DiscriminativeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1)

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
