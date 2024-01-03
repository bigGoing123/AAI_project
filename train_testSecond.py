import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from testDataSet import testDataSet
from CNN import CNN
from tqdm import tqdm
import torch.nn as nn
from testDataSet import testDataSet
import matplotlib.pyplot as plt
import pandas as pd
torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 5  # 训练整批数据的次数
BATCH_SIZE = 64
LR = 0.001  # 学习率

def delete_error_img():
    """
    由于train数据集中图片噪音过多，现用官方数据训练完成的模型对一些标签错误的图片进行筛选，模型准确率已经在98%左右；
    如果模型预测的标签与实际标签不一致，则删除该图片
    """

    root_dir = './processed_data/train'  # 替换为数据集根目录路径
    dataset = testDataSet(root_dir)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 加载模型

    cnn = CNN()
    # 如果模型已经训练过，确保加载模型权重
    #该模型是用mnist官方数据集训练完成的
    cnn.load_state_dict(torch.load('cnn2.pkl'))
    # 将模型设置为评估模式
    cnn.eval()
    delect_index=[]#需要被删除的下标
    for index, (data, label) in enumerate(tqdm(train_loader)):
        # 模型预测
        output = cnn(data)
        _, predicted = torch.max(output.data, 1)
        # 检查预测是否正确
        batch_start_index = index * train_loader.batch_size
        for idx, pred in enumerate(predicted):
            absolute_idx = batch_start_index + idx  # 计算在整个数据集中的索引
            if pred.item() != label[idx].item():
                delect_index.append(absolute_idx)
    dataset.delete_items(delect_index)
    # 创建一个新的 DataLoader用来训练
    new_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return new_dataloader


def train_for_new_model(train_loader):
    """
    dataloader是要训练的数据集
    通过cnn训练已经清理完成的数据集，来得到一个针对于该数据集的新模型。
    """
    cnn2 = CNN()
    # 优化器选择Adam
    optimizer = torch.optim.Adam(cnn2.parameters(), lr=LR)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
    # 开始训练
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # 分配batch data
            output = cnn2(b_x)  # 先将数据放到cnn中计算output
            loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度

            # if step % 50 == 0:
            #     test_output = cnn2(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.numpy()
            #     accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            #     print('Epoch: ', int(epoch), '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    torch.save(cnn2.state_dict(), 'cnnLast.pkl')#保存新模型
if __name__ == '__main__':
    train_loader=delete_error_img()
    train_for_new_model(train_loader)
    # 加载模型
    cnn2 = CNN()
    # 如果模型已经训练过，确保加载模型权重
    cnn2.load_state_dict(torch.load('cnnLast.pkl'))
    # 将模型设置为评估模式
    cnn2.eval()
    #将一个文件夹中的所有文件名写入到一个numpy数组中
    folder_path = './processed_data/test/'
    target_folder = './png_images/new/'
    test_dataset = testDataSet(folder_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    results =[]
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):

            # 获取预测结果
            outputs = cnn2(data)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())
    # 处理或保存测试结果
    file_names=np.array([str(f)+".npy" for f in range(9900)])
    predictions=pd.DataFrame({'fileNames':file_names,'predication':results})
    predictions.to_csv('predictionsLast.csv',index=False)