

import torch
from torch.utils.data import Dataset, DataLoader


class ShakespeareDataset(Dataset):
    def __init__(self, data, sequence_length):
        """
        Custom dataset object designed to pass strings of text to RNN training and testing loop

        Args:
            data: string of text to be sampled from
            sequence_length: length of each sequence generated
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length - 1

    def __getitem__(self, idx):
        input = self.data[idx : idx + self.sequence_length]
        target = self.data[idx + 1 : idx + self.sequence_length + 1]        # Shifted by 1
        return input, target