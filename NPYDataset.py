import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class NPYDataset(Dataset):
    def __init__(self, folder_path):
        self.file_names = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        npy_path = self.file_names[idx]
        npy_data = np.load(npy_path)
        npy_data = np.sum(npy_data, axis=0)
        return torch.from_numpy(npy_data).unsqueeze(0).float()
