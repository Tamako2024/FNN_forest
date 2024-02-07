import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from FNN import FeedForwardNN as FNN
from tqdm import tqdm
import copy
import os


def load_and_preprocess_data(filepath, n=None):
    """
    :param filepath: 训练数据的TXT
    :param n: 取得行数  n为None则读取所有行
    """
    # 读取TXT文件的数据，只读取前n行，列分隔符为逗号
    data = pd.read_csv(filepath, header=None, sep=',', nrows=n).astype(np.float64)
    # Scale数据
    data.iloc[:, :12] *= 0.0001  # Scale前12列
    data.iloc[:, 12] *= 0.01  # Scale最后一列
    # 剔除异常值
    data = data[
        (data.iloc[:, :12] >= 0).all(axis=1) & (data.iloc[:, :12] <= 1).all(axis=1) & (data.iloc[:, 12] >= 0) & (
                data.iloc[:, 12] <= 100)]
    # 转换为numpy数组
    data = data.to_numpy()
    return data


def split_data(input_data, target_labels, use_split=None):
    """
    :param input_data: 输入特征数据
    :param target_labels: 目标标签数据
    :param use_split: 一个整数，表示是否拆分数据集（None=不拆分，所有数据为训练集；True=拆分），默认不拆分
    :return: 拆分好的数据集
    """
    # 创建数据集
    dataset = TensorDataset(input_data, target_labels)

    if use_split:
        # 拆分数据集
        train_size = int(len(dataset) * 0.7)
        val_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    else:
        # 不拆分数据集，所有数据用于训练
        train_dataset = dataset
        val_dataset = None
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train(filepath, num_epochs, batch_size, path="", use_split=None):
    """
    :param filepath: 训练数据TXT
    :param num_epochs: epochs
    :param batch_size: batch
    :param path: weights保存文件夹
    :param use_split: 是否拆分数据集（None=不拆分，所有数据为训练集；True=拆分，7：2：1划分数据），默认不拆分
    :return:
    """
    # 创建模型实例
    model = FNN()

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 数据加载和预处理
    data = load_and_preprocess_data(filepath)
    input_data = torch.tensor(data[:, :12], dtype=torch.float32)
    target_labels = torch.tensor(data[:, 12], dtype=torch.float32).view(-1, 1)
    train_dataset, val_dataset, test_dataset = split_data(input_data, target_labels, use_split)
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

    # 初始化保存最佳模型的变量
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') if use_split else None  # 如果有验证集才算

    # 训练模型
    with open(os.path.join(path, 'logs.txt'), 'w') as loss_file:
        loss_file.write('epoch,train_loss,val_loss\n')
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', leave=True)
            for inputs, labels in train_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # 如果有验证集，进行验证阶段
            if use_split:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()
                val_loss /= len(val_loader)

                # 输出每个epoch的train_loss和val_loss
                tqdm.write(f'\nTrain Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                # 记录最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(path, 'best.pt'))

                # 记录log
                loss_file.write(f'{epoch},{train_loss},{val_loss}\n')
            else:
                # 如果不拆分数据集，只记录train_loss
                tqdm.write(f'\nTrain Loss: {train_loss:.4f}')
                loss_file.write(f'{epoch},{train_loss},None\n')

    # 如果有测试集, 测试模型
    if use_split:
        model.eval()
        model.load_state_dict(best_model_wts)
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), os.path.join(path, 'last.pt'))


if __name__ == "__main__":
    output = 'train_data.txt'
    train(output, num_epochs=1, batch_size=32)
