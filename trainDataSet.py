import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class trainDataSet(Dataset):
    def __init__(self, root_dir):
        """
        初始化数据集
        :param root_dir: 包含所有子文件夹（0-9）的根目录
        """
        self.data_files = []
        self.labels = []
        # 遍历根目录下的每个文件夹
        for label in range(10):  # 假设标签是从0到9
            folder_path = os.path.join(root_dir, str(label))
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.npy'):
                        file_path = os.path.join(folder_path, file)
                        # 保存文件路径而不是加载数据
                        self.data_files.append(file_path)
                        # 使用文件夹名称作为标签
                        self.labels.append(label)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 按需加载数据
        data = np.load(self.data_files[idx])
        data = np.sum(data, axis=0)
        label = self.labels[idx]
        return torch.from_numpy(data).unsqueeze(0).float(), label

    def get_correct_data(self, indices):
        self.data_files = [d for i, d in enumerate(self.data_files) if i in indices]
        self.labels = [l for i, l in enumerate(self.labels) if i in indices]
