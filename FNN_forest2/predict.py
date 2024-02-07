from pyhdf.SD import SD, SDC
import torch
from tqdm import tqdm
import os
import numpy as np
from FNN import FeedForwardNN as FNN


def predict(input_file, weights, labels=None, output_path="", block_size=1000, nan_num=0):
    """
    :param input_file: 被预测的hdf4名称
    :param output_path: 预测结果hdf4文件夹路径
    :param  weights: 模型权重
    :param labels: HDF4文件中输入12维特征向量的标签列表。
    :param block_size: 分块读取hdf并预测，每一块的大小
    :param nan_num: 无效值
    :return:
    """
    if labels is None:
        labels = [
            'MonthNDVI00', 'MonthNDVI01', 'MonthNDVI02', 'MonthNDVI03',
            'MonthRefl00B01', 'MonthRefl01B01', 'MonthRefl02B01', 'MonthRefl03B01',
            'MonthRefl00B07', 'MonthRefl01B07', 'MonthRefl02B07', 'MonthRefl03B07'
        ]

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练模型
    model = FNN().to(device)
    model_state_dict = torch.load(weights, map_location=device)
    model.load_state_dict(model_state_dict)

    model.eval()  # 确保模型处于评估模式

    # 打开输入文件
    file_A = SD(input_file, SDC.READ)

    # 获取输入数据集的形状
    datasets = {label: file_A.select(label) for label in labels}
    shape = datasets[labels[0]].info()[2]  # 假设所有数据集形状相同

    # 创建输出文件和TreeCover数据集
    file_name = os.path.basename(input_file).replace('MODFeatureQKMV2', 'GLOBMAPFTC')

    output_file = os.path.join(output_path, file_name)
    file_B = SD(output_file, SDC.WRITE | SDC.CREATE)
    tree_cover_ds = file_B.create('TreeCover', SDC.INT16, shape)  # 这里设置输出的HDF4数据类型

    # 使用tqdm显示进度
    num_blocks_i = (shape[0] + block_size - 1) // block_size
    num_blocks_j = (shape[1] + block_size - 1) // block_size
    total_blocks = num_blocks_i * num_blocks_j
    progress_bar = tqdm(total=total_blocks, desc='Processing blocks')

    # 分块读取数据并进行预测
    for i in range(0, shape[0], block_size):
        for j in range(0, shape[1], block_size):
            # 确定实际的块大小
            actual_block_size_i = min(block_size, shape[0] - i)
            actual_block_size_j = min(block_size, shape[1] - j)

            # 从输入文件中读取数据集
            data_blocks = []
            for label in labels:
                dataset = datasets[label]
                data_block = dataset[i:i + block_size, j:j + block_size] * 0.0001  # 归一化
                data_blocks.append(torch.from_numpy(data_block).to(device))

            # 将数据转换为张量并在适当的轴上堆叠以匹配模型输入
            input_tensor = torch.stack(data_blocks, dim=-1).view(-1, len(labels)).float()

            # 使用模型进行预测
            with torch.no_grad():
                output = model(input_tensor)

            # 计算输出块的实际大小
            actual_block_size_i = min(block_size, shape[0] - i)
            actual_block_size_j = min(block_size, shape[1] - j)
            actual_block_total_size = actual_block_size_i * actual_block_size_j

            # 将预测结果处理后写入输出文件
            predicted_block = output.cpu().numpy()[:actual_block_total_size] * 100
            predicted_block = predicted_block.astype(np.int16)  # 将预测结果转换为int16
            predicted_block = np.clip(predicted_block, 0, 10000)  # 将值限制在0和100之间
            predicted_block = predicted_block.reshape(actual_block_size_i, actual_block_size_j)

            tree_cover_ds[i:i + actual_block_size_i, j:j + actual_block_size_j] = predicted_block
            progress_bar.update(1)

    # 关闭文件
    progress_bar.close()
    file_A.end()
    file_B.end()


def batch_predict(txt_file, input_path, weights, output_path, block_size=1000, nan_num=0):
    """
    :param txt_file: 包含HDF4文件名称列表的TXT文件路径
    :param input_path: TXT对应的hdf4文件夹路径
    :param weights: 模型权重文件路径
    :param output_path: 预测结果hdf4文件夹路径
    :param block_size: 分块读取hdf并预测，每一块的大小
    :param nan_num: 无效值
    """
    # 打开TXT文件并读取HDF4文件名称
    with open(txt_file, 'r') as file:
        hdf4_filenames = file.readlines()

    # 移除文件名字符串中的空白字符
    hdf4_filenames = [filename.strip() for filename in hdf4_filenames]

    # 遍历列表，对每个HDF4文件进行预测
    for filename in hdf4_filenames:
        hdf4_file_path = os.path.join(input_path, filename)  # 构造完整的文件路径
        print(f"Processing {hdf4_file_path} ...")
        predict(hdf4_file_path, weights, output_path=output_path, block_size=block_size, nan_num=nan_num)


if __name__ == "__main__":
    txt_file = 'list.txt'  # TXT文件路径
    input_path = r'C:\Users\Tamako\Desktop\MODFeatureQKMV2.A2020001.h27v05'  # HDF4文件夹路径
    weights = 'best.pt'  # 模型权重文件路径
    output_path = ''  # 预测结果的输出目录
    batch_predict(txt_file, input_path, weights, output_path)
