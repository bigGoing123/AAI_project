import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class NPYDataset(Dataset):
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


root_dir = './processed_data/train-modify'  # 替换为您的数据集根目录路径
dataset = NPYDataset(root_dir)
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)

from CNN import CNN

batch_size = 64
# 加载模型

cnn = CNN()
# 如果模型已经训练过，确保加载模型权重
cnn.load_state_dict(torch.load('cnn2.pkl'))
# 将模型设置为评估模式
cnn.eval()
from tqdm import tqdm

for index, (data, label) in enumerate(tqdm(trainloader)):
    # 模型预测

    output = cnn(data)
    _, predicted = torch.max(output.data, 1)
    # 检查预测是否正确
    batch_start_index = index * trainloader.batch_size

    for idx, pred in enumerate(predicted):
        absolute_idx = batch_start_index + idx  # 计算在整个数据集中的索引
        if pred.item() != label[idx].item():
            file_path = dataset.data_files[absolute_idx]
            if os.path.exists(file_path):  # 确保文件存在
                os.remove(file_path)