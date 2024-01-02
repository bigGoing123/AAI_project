import torch
import numpy as np
import os
from tqdm import tqdm
from CNN import CNN
from NPYDataset import NPYDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
# 加载.npy文件
batch_size = 64  # 根据您的硬件调整这个数值
# 加载模型
cnn = CNN()
# 如果模型已经训练过，确保加载模型权重
cnn.load_state_dict(torch.load('cnn2.pkl'))
# 将模型设置为评估模式
cnn.eval()
#将一个文件夹中的所有文件名写入到一个numpy数组中
folder_path = './mnist/processed_data/test/'
target_folder = './png_images/'
test_dataset = NPYDataset(folder_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
results =[]
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):

        # 获取预测结果
        outputs = cnn(data)
        _, predicted = torch.max(outputs, 1)
        # 遍历批次中的每个图像及其预测
        for i, (image, pred) in enumerate(zip(data, predicted)):
            # 处理图像数据
            image_np = image.squeeze().numpy()  # 假设图像是单通道的

            # 构建文件名（包含预测结果）
            file_name = f"batch{batch_idx}_img{i}_pred{pred.item()}.png"

            # 保存图像
            plt.imsave(os.path.join(target_folder, file_name), image_np, cmap='gray')

        results.extend(predicted.cpu().numpy())


# 处理或保存测试结果
file_names=np.array([str(f)+".npy" for f in range(9900)])
predictions=pd.DataFrame({'fileNames':file_names,'predication':results})
predictions.to_csv('predictions.csv',index=False)
