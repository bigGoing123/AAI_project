import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from CNN import CNN
from tqdm import tqdm
import torch.nn as nn
from trainDataSet import trainDataSet
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同
from testDataSet import testDataSet
# 超参数
EPOCH = 5 # 训练整批数据的次数
BATCH_SIZE = 64
LR = 0.001  # 学习率

def delete_error_img(root_dir):
    """
    由于train数据集中图片噪音过多，现用官方数据训练完成的模型对一些标签错误的图片进行筛选，模型准确率已经在98%左右；
    如果模型预测的标签与实际标签不一致，则删除该图片
    """

    # root_dir = './processed_data/train'  # 替换为数据集根目录路径
    dataset = trainDataSet(root_dir)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 加载模型

    cnn = CNN()
    # 如果模型已经训练过，确保加载模型权重
    # 该模型是用mnist官方数据集训练完成的
    cnn.load_state_dict(torch.load('cnn2.pkl'))
    # 将模型设置为评估模式
    cnn.eval()
    correct_index = []  # 这是正确的绝对index
    for index, (data, label) in enumerate(tqdm(train_loader)):
        # 模型预测
        output = cnn(data)
        _, predicted = torch.max(output.data, 1)
        # 检查预测是否正确
        batch_start_index = index * train_loader.batch_size
        for idx, pred in enumerate(predicted):
            absolute_idx = batch_start_index + idx  # 计算在整个数据集中的索引
            if pred.item() == label[idx].item():
                correct_index.append(absolute_idx)
                # correct_file_label.append(pred.item())

    print("删除噪音数据中........大概2min")
    dataset.get_correct_data(correct_index)
    # dataset.set_labels(correct_file_label)
    # 创建一个新的 dataset
    new_dataset = train_loader.dataset
    return new_dataset


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
        for step, (b_x, b_y) in enumerate(tqdm(train_loader)):  # 分配batch data
            output = cnn2(b_x)  # 先将数据放到cnn中计算output
            loss = loss_func(output, b_y)  # 输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度
    torch.save(cnn2.state_dict(), 'cnnLast.pkl')#保存新模型
def get_img_from_npy(train_loader,target_dir,model_file):
    """
    :param train_loader: 传入一个dataloader，可以带标签也可以不带
    :param target_dir:目标文件夹
    :param cnn:训练好的模型，用来预测该npy文件的数字
    :return: None
    将传入的loader中所有的npy文件，在目标文件夹下保存为相应的图片
    """
    cnn = CNN()
    # 如果模型已经训练过，确保加载模型权重
    cnn.load_state_dict(torch.load(model_file))
    # 将模型设置为评估模式
    cnn.eval()
    os.makedirs(target_dir, exist_ok=True)
    # total_samples=[]
    # correct_predictions=[]
    k=0
    for data in tqdm(train_loader):
        outputs = cnn(data)
        _, predicted = torch.max(outputs, 1)
        # total_samples += labels.size(0)
        # correct_predictions += (predicted == labels).sum().item()
        for i in range(data.size(0)):  # 遍历批次中的每个图像
            image_tensor = data[i]  # 获取单个图像张量，形状为 [channels, height, width]
            k += 1
            image = transforms.ToPILImage()(image_tensor.squeeze(0))
            filename = f".//{target_dir}//{k}_{predicted[i]}.png"
            image.save(filename)
def get_test_results_tocsv(test_loader,model_file,target_dir):
    """

    :param test_loader: 加载一个dataloader，不包含labels
    :param model_file: 训练好的模型文件
    :param target_dir: 将最后的结果保存在目标文件夹下
    :return: None
    """
    os.makedirs(target_dir, exist_ok=True)
    # 加载模型
    cnn = CNN()
    # 如果模型已经训练过，确保加载模型权重
    cnn.load_state_dict(torch.load(model_file))
    # 将模型设置为评估模式
    cnn.eval()
    # 将一个文件夹中的所有文件名写入到一个numpy数组中
    results = []
    with torch.no_grad():
        for index, data in enumerate(tqdm(test_loader)):
            # 获取预测结果
            outputs = cnn(data)
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())
    # 处理或保存测试结果
    file_names = np.array([str(f) + ".npy" for f in range(9900)])
    predictions = pd.DataFrame({'fileNames': file_names, 'predication': results})
    predictions.to_csv(f'{target_dir}/predictionsLast.csv', index=False)
if __name__ == '__main__':
    #需要保证当前路径下有processed_data文件夹
    root_dir = './processed_data/train'  # 替换为数据集根目录路径
    train_set=delete_error_img(root_dir)
    train_loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
    print("开始训练新模型.........")
    train_for_new_model(train_loader)
    print("训练结束..........")
    # 加载模型

    #将一个文件夹中的所有文件名写入到一个numpy数组中
    folder_path = './processed_data/test/'
    # target_folder = './png_images/new/'
    test_dataset = testDataSet(folder_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #将 test文件的图片保存在img文件夹下
    get_img_from_npy(test_loader,'./test_results/img','cnnLast.pkl')
    #将result保存在csv文件夹中
    get_test_results_tocsv(test_loader,'cnnLast.pkl','./test_results/')