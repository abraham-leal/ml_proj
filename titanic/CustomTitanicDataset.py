from torch.utils.data import Dataset
import pandas as pd
import torch


class CustomTitanicDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.data: pd.DataFrame
        self.data = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]

        datapoint = torch.tensor(row.iloc[1:7].values, dtype=torch.float32)
        label = torch.tensor(row.iloc[0], dtype=torch.float32)

        if self.transform:
            dp = self.transform(datapoint)
        if self.target_transform:
            label = self.target_transform(label)

        return datapoint, label
