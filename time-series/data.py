import numpy as np
import torch
from torch.utils.data import Dataset


def extract_time (data):
    """Returns Maximum sequence length and each sequence length.

    Args:
        - data: original data

    Returns:
        - time: extracted time information
        - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))

    return time, max_seq_len


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


class DiscriminativeDataset(Dataset):
    def __init__(self, real, synthetic):
        self.X = torch.cat((torch.tensor(real).float(), torch.tensor(synthetic).float()))
        self.y = torch.cat((torch.full((real.shape[0],), 1.0), torch.full((synthetic.shape[0],), 0.0)))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class RealData:
    def __init__(self, data_path):
        self.data_path = data_path

    def real_data_loading(self, data_name, seq_len):
        """Load and preprocess real-world datasets.

        Args:
            - data_name: stock or energy
            - seq_len: sequence length

        Returns:
            - data: preprocessed data.
        """
        assert data_name in ['stock', 'energy']

        if data_name == 'stock':
            ori_data = np.loadtxt(self.data_path + '/data/stock_data.csv', delimiter=",", skiprows=1)
        else:
            ori_data = np.loadtxt(self.data_path + '/data/energy_data.csv', delimiter=",", skiprows=1)

        # Flip the data to make chronological data
        ori_data = ori_data[::-1]
        # Normalize the data
        ori_data = MinMaxScaler(ori_data)

        # Preprocess the dataset
        temp_data = []
        # Cut data by sequence length
        for i in range(0, len(ori_data) - seq_len):
            _x = ori_data[i:i + seq_len]
            temp_data.append(_x)

        # Mix the datasets (to make it similar to i.i.d)
        idx = np.random.permutation(len(temp_data))
        data = []
        for i in range(len(temp_data)):
            data.append(temp_data[idx[i]])

        return data


class SequenceData(Dataset):
    def __init__(self, features):
        self.X = torch.tensor(features).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i]


def train_test_divide(data_x, data_t, train_rate=0.8):
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*train_rate)]
    test_idx = idx[int(no*train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    return np.asarray(train_x), np.asarray(test_x), np.asarray(train_t), np.asarray(test_t)
