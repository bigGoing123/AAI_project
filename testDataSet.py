import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
class testDataSet(Dataset):
    """
    传入源文件夹地址，该文件夹下直接是所有的npy文件
    """
    def __init__(self, root_dir):
        """
        初始化数据集
        :param root_dir: 包含所有子文件夹（0-9）的根目录
        """
        self.data_files = []
        for i in range(9900):
            self.data_files.append(os.path.join(root_dir, str(i) + '.npy'))
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 按需加载数据
        data = np.load(self.data_files[idx])
        data = np.sum(data, axis=0)
        return torch.from_numpy(data).unsqueeze(0).float()
