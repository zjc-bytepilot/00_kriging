import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import os


def split_and_repeat_channels(image):
    """
    将一个形状为 (H, W, 4) 的 NumPy 图像返回形状为 (4, 1, H, W) 的张量。

    参数:
    image (numpy.ndarray): 输入的形状为 (H, W, 4) 的图像数组。

    返回:
    Tensor: 形状为 (4, 1, H, W) 的张量，表示复制后的通道。
    """
    # 确保输入图像是 (H, W, 4)
    if image.shape[2] != 4:
        raise ValueError("Input image must have 4 channels.")
    # 将 NumPy 数组转换为 torch.Tensor，并确保数据类型是 float32
    image_tensor = torch.tensor(image, dtype=torch.float32)
    # 调整维度顺序，从 (H, W, C) 到 (C, H, W)，即 (4, H, W)
    image_tensor = image_tensor.permute(2, 0, 1)
    # 将每个通道扩展为 (1, H, W) 的形状，并添加到一个列表中
    result = image_tensor.unsqueeze(1)  # 在第一个维度上增加一个维度，得到 (4, 1, H, W)
    return result

def l1_loss(pred, target):
    """
    计算L1损失 (即绝对误差)
    :param pred: 预测矩阵，大小为 (batch_size, C, H, W)
    :param target: 真实矩阵，大小为 (batch_size, C, H, W)
    :return: L1 损失
    """
    loss = torch.abs(pred - target)  # 计算每个元素的绝对误差
    return loss.mean()  # 计算平均损失


def l2_loss(pred, target):
    """
    计算L2损失 (即均方误差)
    :param pred: 预测矩阵，大小为 (batch_size, C, H, W)
    :param target: 真实矩阵，大小为 (batch_size, C, H, W)
    :return: L2 损失
    """
    loss = (pred - target) ** 2  # 计算每个元素的平方误差
    return loss.mean()  # 返回均值作为损失


def Min_var_loss(outputs, matrix_right):
    # 按元素相乘
    product = outputs[:, :-2, :] * matrix_right[:, :-2, :]  # 逐元素相乘，结果形状为 (4, 90, 9)
    # 沿第二个维度（92）求和，得到形状为 (4, 1, 9)
    # result = torch.abs(product).sum(dim=1, keepdim=True)  # 沿第1维求和，保持维度 (4, 1, 9)
    result = product.sum(dim=1, keepdim=True)  # 沿第1维求和，保持维度 (4, 1, 9)
    # 将最后两个元素的 `matrix_right` 加入结果中
    result1 = result + matrix_right[:, -2, :]
    # 沿第三个维度（9）求和或求平均，得到形状为 (4, 1, 1)
    loss = result1.mean()  # 沿第2维求平均，保持维度
    # 返回标量损失
    return loss.squeeze()  # 去掉多余的维度，返回一个标量


def co_conv_module(kernels, x, y):
    """
    该函数实现对两个输入张量 x 和 y 使用不同的卷积核进行卷积操作，并进行相应的输出降尺度处理。

    :param kernels: 卷积核张量，形状为 [batch_size, input_channels, kernel_height, kernel_width]。
    :param x: 第一个输入张量，形状为 [batch_size, channels, height, width]。
    :param y: 第二个输入张量，形状为 [batch_size, channels, height, width]。
    :return: 返回卷积操作后的结果张量，经过处理的 x 和 y 的卷积输出。
    """
    result_x = torch.zeros(4, 1, 30, 30)  # 初始化 result_x 张量，存储第一个卷积操作结果
    result_y = torch.zeros(4, 1, 30, 30)  # 初始化 result_y 张量，存储第二个卷积操作结果
    # 将 kernels 分为两个部分，一个用于 x 的卷积操作，另一个用于 y 的卷积操作
    kernels_x = kernels[:, 0:9, :].permute(0, 2, 1).reshape(4, 9, 3, 3)
    kernels_y = kernels[:, 9:-2, :].permute(0, 2, 1).reshape(4, 9, 9, 9)  # 将 kernels 的剩余部分提取并重塑为 (4, 9, 9, 9)
    # 遍历每个样本（批次维度）
    for k in range(x.size(0)):  # 遍历 batch_size 维度
        outputs_x = []  # 存储 x 的卷积结果
        outputs_y = []  # 存储 y 的卷积结果
        # 遍历每个通道（假设通道数为 9）
        for i in range(x.size(1)):  # x.size(1) == 9 个通道
            # 取出 x 和 y 的第 i 个通道的卷积核，调整为 [1, 1, kernel_height, kernel_width] 的形状
            kernel_x = kernels_x[k][i].unsqueeze(0).unsqueeze(0)  # 调整 kernel_x 的形状
            kernel_y = kernels_y[k][i].unsqueeze(0).unsqueeze(0)  # 调整 kernel_y 的形状
            # 对 x 的第 i 个通道进行卷积操作
            out_x = F.conv2d(
                x[k, i:i + 1, :, :].unsqueeze(0),  # 当前样本的第 i 个通道
                kernel_x,  # 当前的卷积核
                stride=1,  # 卷积步幅为 1
                padding=1,  # 填充为 1
                bias=None  # 不使用偏置
            )
            # 对 y 的第 i 个通道进行卷积操作
            out_y = F.conv2d(
                y[k, i:i + 1, :, :].unsqueeze(0),  # 当前样本的第 i 个通道
                kernel_y,  # 当前的卷积核
                stride=3,  # 卷积步幅为 3
                padding=4,  # 填充为 4
                bias=None  # 不使用偏置
            )
            outputs_x.append(out_x)  # 将 x 的卷积结果添加到 outputs_x 中
            outputs_y.append(out_y)  # 将 y 的卷积结果添加到 outputs_y 中
        # 在通道维度上拼接所有通道的卷积结果，拼接后的张量形状为 [1, 9, height, width]
        out_x = torch.cat(outputs_x, dim=1)
        out_y = torch.cat(outputs_y, dim=1)
        # 对卷积结果进行降尺度处理，调用 Output_Processing 函数（这里假设该函数处理降尺度逻辑）
        out_x = Output_Processing(out_x)
        out_y = Output_Processing(out_y)
        # 将处理后的卷积结果存储到初始化的 result_x 和 result_y 中
        result_x[k, :, :, :] = out_x[0, :, :, :]
        result_y[k, :, :, :] = out_y[0, :, :, :]
    # 返回处理后的卷积结果的和，result_x 和 result_y 代表了两个输入的卷积结果
    return result_x + result_y


def Output_Processing(out):
    a, b, c, d = out.shape  # 获取卷积结果的形状
    out = out.reshape(1, 9, c * d)  # 展平每个 c x d 矩阵为一个 c*d 长度的向量
    out = out.permute(0, 2, 1)  # 转置通道和展平维度
    out = out.reshape(1, c * d, 3, 3)  # 重塑为每个像素点的 3x3 块
    out = out.reshape(1, c, d, 3, 3)  # 恢复每个像素块的原始排列
    out = out.permute(0, 1, 3, 2, 4)  # 调整维度顺序
    out = out.reshape(1, 3 * c, 3 * d)  # 拼接形成更大的分辨率
    out = out.reshape(1, 1, 3 * c, 3 * d)  # 恢复为单通道
    return out

def _load_mat_arrays(data_name, path, mat_name, report_missing=False):
    """从一组 MAT 文件中读取同名数组，统一转换为 float32。"""
    loaded_images = []
    for date in data_name:
        # 保留项目现有的文件命名规则：path + date + '.mat'。
        file_path = f"{path}{date}.mat"
        if not os.path.exists(file_path):
            if report_missing:
                print(f"文件 {file_path} 不存在！")
            continue

        image_dict = scipy.io.loadmat(file_path)
        image_data = np.asarray(image_dict[mat_name], dtype=np.float32)
        loaded_images.append((file_path, image_data))
    return loaded_images


def load_data(data_name, path, mat_name):
    """
    加载指定路径下的 `.mat` 文件，并将其中的图像数据提取、处理后返回。

    参数:
    - data_name (list): 包含日期或标识的列表，用于构造文件路径。
    - path (str): `.mat` 文件的目录路径。
    - mat_name (str): `.mat` 文件中目标数据的键名。

    返回:
    - image_list (list): 包含处理后图像的张量列表，每个元素为 torch.Tensor。
    """
    loaded_images = _load_mat_arrays(data_name, path, mat_name, report_missing=True)
    image_tensors = []
    for file_path, image_data in loaded_images:
        image_tensor = split_and_repeat_channels(image_data)
        image_tensors.append(image_tensor)
        print(f"Image shape for {file_path}: {image_tensor.shape}")
    return image_tensors


def load_data_numpy(data_name, path, mat_name):
    """
    加载指定路径下的 `.mat` 文件，并将其中的图像数据提取、处理后返回。

    参数:
    - data_name (list): 包含日期或标识的列表，用于构造文件路径。
    - path (str): `.mat` 文件的目录路径。
    - mat_name (str): `.mat` 文件中目标数据的键名。

    返回:
    - image_list (list): 包含处理后图像的张量列表，每个元素为 torch.Tensor。
    """
    return [image_data for _, image_data in _load_mat_arrays(data_name, path, mat_name)]


def evaluate_relation_spectral(realdata, predictdata, s):
    """
    评估预测数据与真实数据的光谱关系指标，包括 RMSE, CC, ERGAS, UIQI 和 SAM。

    参数:
    realdata (numpy.ndarray): 实际的光谱数据，形状为 (高度, 宽度, 波段数)。
    predictdata (numpy.ndarray): 预测的光谱数据，形状同 `realdata`。
    s (float): 数据的空间分辨率比，用于计算 ERGAS 指标。

    返回:
    tuple: 包括以下指标的数组:
        - RMSE: 各波段的均方根误差和平均值。
        - CC: 各波段的相关系数和平均值。
        - ERGAS: 光谱误差的综合评估指标。
        - UIQI: 各波段的图像质量指数和平均值。
        - SAM: 光谱角度映射值。
    """
    # 提取数据维度
    a, b, c = realdata.shape
    # 初始化结果存储列表
    RMSE0 = []  # 每个波段的 RMSE
    CC0 = []  # 每个波段的相关系数 CC
    ERGAS0 = []  # 每个波段的 ERGAS
    UIQI0 = []  # 每个波段的 UIQI
    # 遍历每个波段计算指标
    for i in range(c):
        # 提取当前波段的真实数据和预测数据
        P = predictdata[:, :, i]
        R = realdata[:, :, i]
        # 计算 RMSE (Root Mean Square Error)
        RMSE1 = np.sum((R - P) ** 2)  # 均方误差
        RMSE0.append(np.sqrt(RMSE1 / (a * b)))  # 开平方并归一化
        # 计算 CC (Correlation Coefficient)
        C_1 = np.sum(P * R) - a * b * np.mean(P) * np.mean(R)
        C_2 = np.sum(P ** 2) - a * b * np.mean(P) ** 2
        C_3 = np.sum(R ** 2) - a * b * np.mean(R) ** 2
        CC0.append(C_1 / np.sqrt(C_2 * C_3))  # 根据公式计算相关系数
        # 计算 ERGAS (Error in Reflectance and Absorption Spectra)
        ERGAS0.append(RMSE0[i] / np.mean(R))
        # 计算 UIQI (Universal Image Quality Index)
        UIQI_1 = 4 * np.mean(P) * np.mean(R) * C_1
        UIQI_2 = (np.mean(P) ** 2 + np.mean(R) ** 2) * (C_2 + C_3)
        UIQI0.append(UIQI_1 / UIQI_2)  # 根据公式计算 UIQI
    # 将每个波段的 RMSE、CC 和 UIQI 指标计算平均值并添加到结果
    RMSE = np.array(RMSE0 + [np.mean(RMSE0)])  # 各波段的 RMSE 和其平均值
    CC = np.array(CC0 + [np.mean(CC0)])  # 各波段的 CC 和其平均值
    UIQI = np.array(UIQI0 + [np.mean(UIQI0)])  # 各波段的 UIQI 和其平均值
    # 计算综合 ERGAS 指标
    ERGAS = 100 * np.linalg.norm(ERGAS0) / (s * np.sqrt(c))
    # 计算 SAM (Spectral Angle Mapper)
    SAM0 = np.zeros((a, b))  # 用于存储每个像素的光谱角度值
    for i in range(a):
        for j in range(b):
            # 提取像素的光谱向量
            VP = predictdata[i, j, :].flatten()
            VR = realdata[i, j, :].flatten()
            # 计算光谱角度映射值，防止零向量导致除零错误
            SAM0[i, j] = np.dot(VP, VR) / (np.linalg.norm(VP) + 1e-7) / (np.linalg.norm(VR) + 1e-7)
    # 计算平均的光谱角度 (单位: 弧度)
    SAM = np.arccos(np.mean(SAM0))
    # 返回所有计算结果
    return RMSE, CC, ERGAS, UIQI, SAM

