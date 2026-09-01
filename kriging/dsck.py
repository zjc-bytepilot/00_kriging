import numpy as np
from numba import jit
from scipy.optimize import least_squares
from .systems import DSCKSystemBuilder, KrigingSolver
from .variogram import CrossVariogramEstimator, VariogramEstimator


# 当前数据集的固定空间尺度。集中定义，避免在多个流程中重复魔法数字。
COARSE_SCALE = 3
FINE_SCALE = 2

@jit(nopython=True)
def D2_D3(hyp_data2, line, column):
    """
    从二维数据转为三维数据。
    Args:
        hyp_data2 (numpy.ndarray): 输入的二维数据，形状为 (dim, nn)
        line (int): 输出的二维数据的行数
        column (int): 输出的二维数据的列数
    Returns:
        numpy.ndarray: 输出的三维数据，形状为 (line, column, dim)
    """

    dim, nn = hyp_data2.shape  # 获取输入数据的维度
    hyp_data3 = np.zeros((line, column, dim))  # 初始化三维数组
    for i in range(dim):
        hyp_data3[:, :, i] = np.reshape(hyp_data2[i, :], (column, line)).T  # 对每一维数据进行reshape操作
    return hyp_data3


def D3_D2(hyp_data3):
    """
    将三维numpy数组转换为二维数组，先通过转置将维度调整，再进行展平操作。
    参数:
    hyp_data3: 输入的三维数组，形状为 (dim1, dim2, dim3)
    返回:
    hyp_data2: 转换后的二维数组，形状为 (dim3, dim1 * dim2)
    """

    # 转置操作：python与matlab的重排列：一个按列一个按行
    hyp_data3_transpose = np.transpose(hyp_data3, (2, 1, 0))  # 移动维度
    # 转换为二维数组：将后两维展平并合并为一个维度
    hyp_data2 = hyp_data3_transpose.reshape(hyp_data3.shape[-1], -1)  # 将最后两维合并为一个维度
    return hyp_data2


def extend_plane(Z, w):
    """
    扩展二维矩阵 Z，在上下和左右边缘添加由重复列和行组成的边界。
    参数:
    Z: 输入的二维矩阵
    w: 扩展的边界大小
    返回:
    Z_extend: 扩展后的二维矩阵
    """

    # 扩展列：在左右边添加重复的列
    Z_extend1 = np.hstack([np.tile(Z[:, 0].reshape(-1, 1), (1, w)), Z, np.tile(Z[:, -1].reshape(-1, 1), (1, w))])
    # 扩展行：在上下边添加重复的行
    Z_extend = np.vstack([np.tile(Z_extend1[0, :], (w, 1)), Z_extend1, np.tile(Z_extend1[-1, :], (w, 1))])
    return Z_extend


def PSF(s, w, b):
    """
    计算点扩散函数 (PSF)，基于给定的窗口大小 w、点半方差 b 和缩放因子 s。
    参数:
    s: 下采样因子
    w: 窗口大小
    b: 点半方差参数
    返回:
    H: 标准化后的PSF矩阵
    """

    # 初始化一个空的矩阵 H0，用于存储 PSF 值
    H0 = np.zeros(((2 * w + 1) * s, (2 * w + 1) * s))
    # 计算 PSF 的值
    for i in range(1, (2 * w + 1) * s + 1):
        for j in range(1, (2 * w + 1) * s + 1):
            # 计算当前点到矩阵中心的欧几里得距离的平方
            # 将 list 转换为 numpy 数组
            Dis2 = np.linalg.norm(
                np.array([i - 0.5, j - 0.5]) - np.array([(2 * w + 1) * s / 2, (2 * w + 1) * s / 2])) ** 2
            H0[i - 1, j - 1] = np.exp(-Dis2 / (2 * b ** 2))  # 根据高斯函数计算 PSF
    # 归一化处理，确保PSF矩阵的和为1
    Hsum = np.sum(H0)
    H = H0 / Hsum
    return H


def downsample_plane(plane, s, w, PSF):
    """
    下采样函数，依据MATLAB代码中的逻辑
    :param plane: 输入的二维图像或矩阵
    :param s: 下采样的比例
    :param w: 窗口大小
    :param PSF: 点扩散函数
    :return: 下采样后的图像矩阵
    """

    # 对plane进行扩展，扩展的大小为w*s
    plane = extend_plane(plane,  w * s)
    sizec, sized = plane.shape
    # 初始化输出数组S
    S = np.zeros((sizec // s, sized // s))
    # 循环遍历输入矩阵，按窗口进行采样
    for i in range(w * s, sizec - w * s, s):
        for j in range(w * s, sized - w * s, s):
            m = (i + s - 1) // s
            n = (j + s - 1) // s
            # 获取局部窗口 (2w+1)*s 子像素
            LW = plane[i - w * s:i + w * s + s, j - w * s:j + w * s + s]
            # 对局部窗口与PSF进行卷积（元素乘法然后求和）
            S[m, n] = np.sum(LW * PSF)
    # 剪去扩展区域的部分
    S = S[w: -w, w: -w]
    return S


def dowmsample_cube(cube, s, w, PSF):
    """
    对三维立方体进行下采样，分别处理每个二维切片。
    参数：
    cube: 输入的三维矩阵
    s: 下采样的缩放因子
    w: 窗口大小
    PSF: 点扩散函数（PSF）
    返回：
    S: 下采样后的三维矩阵
    """

    a0, b0, c0 = cube.shape
    S = np.zeros((int(a0 / s), int(b0 / s), c0))  # 初始化输出矩阵，保持和输入矩阵相同的形状
    for k in range(c0):
        S[:, :, k] = downsample_plane(cube[:, :, k], s, w, PSF)
    return S

@jit(nopython=True)
def myfun(x, xdata):
    """
    计算函数 F = x[0] * (1 - exp(-xdata / x[1]))
    参数:
    x: 包含两个元素的数组或列表，其中 x[0] 是系数，x[1] 是用于指数的参数
    xdata: 输入数据，通常是一个 numpy 数组
    返回:
    F: 计算结果，numpy 数组形式
    """

    F = x[0] * (1 - np.exp(-xdata / x[1]))
    return F


def myfun_fit(x, xdata, y_data):
    """
    计算函数 F = x[0] * (1 - exp(-xdata / x[1]))
    参数:
    x: 包含两个元素的数组或列表，其中 x[0] 是系数，x[1] 是用于指数的参数
    xdata: 输入数据，通常是一个 numpy 数组
    返回:
    F: 计算结果，numpy 数组形式
    """

    F = x[0] * (1 - np.exp(-xdata / x[1]))
    return F - y_data

@jit(nopython=True)
def myfun2(x, xdata):
    """
    计算函数 F = x[0] + x[1] * (1 - exp(-xdata / x[2]))
    参数:
    x: 包含三个元素的数组或列表，其中 x[0] 是常数项，x[1] 是系数，x[2] 是用于指数的参数
    xdata: 输入数据，通常是一个 numpy 数组
    返回:
    F: 计算结果，numpy 数组形式
    """

    F = x[0] + x[1] * (1 - np.exp(-xdata / x[2]))
    return F


def myfun2_fit(x, xdata, y_data):
    """
    计算函数 F = x[0] + x[1] * (1 - exp(-xdata / x[2]))
    参数:
    x: 包含三个元素的数组或列表，其中 x[0] 是常数项，x[1] 是系数，x[2] 是用于指数的参数
    xdata: 输入数据，通常是一个 numpy 数组
    返回:
    F: 计算结果，numpy 数组形式
    """

    F = x[0] + x[1] * (1 - np.exp(-xdata / x[2]))
    return F - y_data

@jit(nopython=True)  # 使用 JIT 编译器进行加速
def semivariogram(J, h):
    """
    计算空间自相关（变异函数）
    参数:
    J: 输入的二维矩阵
    h: 延迟距离（即用于计算空间自相关的步长）
    返回:
    rh: 变异函数值
    """

    a, b = J.shape  # 获取矩阵的维度，a 是行数，b 是列数
    N1, r1 = 0, 0
    # 计算r1和N1
    for i in range(h, a):
        for j in range(b):
            r1 += (J[i, j] - J[i - h, j]) ** 2
            N1 += 1
    N2, r2 = 0, 0
    # 计算r2和N2
    for i in range(a - h):
        for j in range(b):
            r2 += (J[i, j] - J[i + h, j]) ** 2
            N2 += 1
    N3, r3 = 0, 0
    # 计算r3和N3
    for i in range(a):
        for j in range(h, b):
            r3 += (J[i, j] - J[i, j - h]) ** 2
            N3 += 1
    N4, r4 = 0, 0
    # 计算r4和N4
    for i in range(a):
        for j in range(b - h):
            r4 += (J[i, j] - J[i, j + h]) ** 2
            N4 += 1
    r = r1 + r2 + r3 + r4  # 所有距离的变异量
    N = N1 + N2 + N3 + N4  # 总数
    rh = r / (2 * N)  # 计算最终的变异函数值
    return rh

@jit(nopython=True)
def semivariogram_cross(M, N, h):
    """
    计算空间自相关（变异函数），输入为两个矩阵 M 和 N，计算它们之间的交叉空间自相关
    参数:
    M: 输入的第一个二维矩阵
    N: 输入的第二个二维矩阵
    h: 延迟距离（即用于计算空间自相关的步长）
    返回:
    rh: 交叉变异函数值
    """

    a, b = M.shape  # 获取矩阵的维度，a 是行数，b 是列数
    # 初始化变异量和总数
    r1, N1 = 0, 0
    r2, N2 = 0, 0
    r3, N3 = 0, 0
    r4, N4 = 0, 0
    # 计算r1和N1
    for i in range(h, a):
        for j in range(b):
            r1 += (M[i, j] - M[i - h, j]) * (N[i, j] - N[i - h, j])
            N1 += 1
    # 计算r2和N2
    for i in range(a - h):
        for j in range(b):
            r2 += (M[i, j] - M[i + h, j]) * (N[i, j] - N[i + h, j])
            N2 += 1
    # 计算r3和N3
    for i in range(a):
        for j in range(h, b):
            r3 += (M[i, j] - M[i, j - h]) * (N[i, j] - N[i, j - h])
            N3 += 1
    # 计算r4和N4
    for i in range(a):
        for j in range(b - h):
            r4 += (M[i, j] - M[i, j + h]) * (N[i, j] - N[i, j + h])
            N4 += 1
    # 总变异量和总数
    r = r1 + r2 + r3 + r4
    N = N1 + N2 + N3 + N4
    # 计算最终的变异函数值
    rh = r / (2 * N)
    return rh

@jit(nopython=True)
def regularization_coarse(h, s0, s, xX):
    """
    相同波段卷积的正则化
    :param h: 滞后距离
    :param s0: 一个参数，影响 Assume_L2 大小
    :param s: 缩放因子
    :param xX: mgfun 的两个参数，假设是一个数组
    :return: raa，正则化结果
    """

    # 创建 Assume_L1 和 Assume_L2
    Assume_L1 = np.zeros((h + 1, 1))
    M1, N1 = np.where(Assume_L1 == 0)
    Assume_L2 = np.zeros((s0, s0))
    M2, N2 = np.where(Assume_L2 == 0)
    raa = np.zeros((h + 1, 1))
    for i in range(h + 1):
        raa[i] = 0
        for m in range(s0 ** 2):
            for n in range(s0 ** 2):
                p1 = np.array([M1[i] * s * s0 + s * M2[m] + 1, N1[i] * s * s0 + s * N2[m] + 1])
                p2 = np.array([M1[0] * s * s0 + s * M2[n] + 1, N1[0] * s * s0 + s * N2[n] + 1])
                raa[i] += myfun(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
    raa /= s0 ** 4
    return raa

@jit(nopython=True)
def deconvolution_coarse(H, s0, s, x_area, Sill_min, Range_min, L_sill, L_range, rate):
    """
    相同波段去正则化/反卷积
    :param H: 滞后距离
    :param s0: 一个参数，影响 Assume_L2 大小
    :param s: 缩放因子
    :param x_area: x_area 用于计算 myfun 和正则化
    :param Sill_min, Range_min: 最小值参数
    :param L_sill, L_range: 长度
    :param rate: 比例因子
    :return: x_best，最好的解
    """

    # 计算 Fa0 向量
    Fa0 = myfun(x_area, np.arange(1, s * s0 * H + 1))
    Fa0_vector = Fa0[s * s0 - 1::s * s0]  # 从 Fa0 中提取特定部分
    Fa0_vector = Fa0_vector.T  # 转置为行向量
    Dif_min = 1e6  # 初始化最小差值
    x_best = None  # 初始化最好的解
    # 遍历不同的 Sill 和 Range
    for i in range(1, L_sill + 1):
        for j in range(1, L_range + 1):
            xp = np.array([(Sill_min + rate * i) * x_area[0], (Range_min + rate * j) * x_area[1]])
            raa0 = regularization_coarse(H, s0, s, xp)
            raa = raa0[1:H + 1] - raa0[0]  # 计算差值
            Dif = np.linalg.norm(raa - Fa0_vector)  # 计算范数（差异）
            if Dif <= Dif_min:
                x_best = xp
                Dif_min = Dif  # 更新最小差异
    return x_best

@jit(nopython=True)
def regularization_cross(h, s0, s, xX):
    """
    相同波段卷积的交叉正则化
    :param h: 滞后距离
    :param s0: 一个参数，影响 Assume_L2 大小
    :param s: 缩放因子
    :param xX: mgfun 的两个参数，假设是一个数组
    :return: raa，正则化结果
    """

    # 初始化 Assume_L1, Assume_L2 和 Assume_L3
    Assume_L1 = np.zeros((h + 1, 1))
    M1, N1 = np.where(Assume_L1 == 0)
    Assume_L2 = np.zeros((s0, s0))
    M2, N2 = np.where(Assume_L2 == 0)
    Assume_L3 = np.zeros((s * s0, s * s0))
    M3, N3 = np.where(Assume_L3 == 0)
    raa = np.zeros((h + 1, 1))
    # 迭代计算 raa
    for i in range(h + 1):
        raa[i] = 0
        for m in range(s0 ** 2):
            for n in range((s * s0) ** 2):
                # 计算 p1 和 p2
                p1 = np.array([M1[i] * s * s0 + M3[n] + 0.5, N1[i] * s * s0 + N3[n] + 0.5])
                p2 = np.array([M1[0] * s * s0 + s * M2[m] + 1, N1[0] * s * s0 + s * N2[m] + 1])
                # 计算 norm(p1 - p2) 并通过 myfun2 更新 raa[i]
                raa[i] += myfun2(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
    # 最后除以 (s0 * s * s0)^2
    raa /= (s0 * s * s0) ** 2
    return raa

@jit(nopython=True)
def deconvolution_cross(H, s0, s, x_area, Constant_min, Sill_min, Range_min, L_sill, L_range, L_constant, rate):
    """
    插值图像的反卷积
    :param H: 滞后距离
    :param s0: 一个参数，影响 Assume_L2 大小
    :param s: 缩放因子
    :param x_area: x_area 用于计算 myfun2 和正则化
    :param Constant_min, Sill_min, Range_min: 最小值参数
    :param L_sill, L_range, L_constant: 长度
    :param rate: 比例因子
    :return: x_best，最好的解
    """

    # 计算 Fa0 向量
    Fa0 = myfun2(x_area, np.arange(s * s0, s * s0  * H + 1, s * s0))
    Fa0_vector = Fa0.T  # 转置为行向量
    Dif_min = 1e6  # 初始化最小差值
    x_best = None  # 初始化最好的解
    # 遍历不同的 Constant, Sill 和 Range
    for m in range(1, L_constant + 1):
        for i in range(1, L_sill + 1):
            for j in range(1, L_range + 1):
                # 计算 xp 参数
                xp = np.array([(Constant_min + rate * m) * x_area[0],
                      (Sill_min + rate * i) * x_area[1],
                      (Range_min + rate * j) * x_area[2]])
                # 计算 raa0，进行正则化
                raa0 = regularization_cross(H, s0, s, xp)
                # 计算差值 raa
                raa = raa0[1:H + 1, 0] - raa0[0, 0]
                # 计算范数差异
                Dif = np.sqrt(np.sum((raa - Fa0_vector) ** 2))
                # 如果差值小于最小差值，更新最优解
                if Dif <= Dif_min:
                    x_best = xp
                    Dif_min = Dif  # 更新最小差异
    return x_best

@jit(nopython=True)
def regularization_fine(h, s, xX):
    """
    执行细化层的正则化计算
    :param h: 滞后距离
    :param s: 缩放因子
    :param xX: mgfun的参数
    :return: raa 数组，包含每个滞后距离的正则化值
    """

    # 假设 Assume_L1 和 Assume_L2 是基于大小创建的零矩阵
    Assume_L1 = np.zeros((h + 1, 1))
    M1, N1 = np.where(Assume_L1 == 0)  # 找到非零元素的索引
    Assume_L2 = np.zeros((s, s))
    M2, N2 = np.where(Assume_L2 == 0)  # 找到非零元素的索引
    raa = np.zeros((h + 1, 1))  # 初始化正则化值数组
    # 遍历每个滞后距离
    for i in range(h + 1):
        raa[i] = 0  # 重置当前滞后距离的正则化值
        # 遍历 Assume_L2 矩阵的每个元素
        for m in range(s**2):
            for n in range(s**2):
                # 计算 p1 和 p2 的位置
                p1 = np.array([M1[i] * s + M2[m] + 0.5, N1[i] * s + N2[m] + 0.5])
                p2 = np.array([M1[0] * s + M2[n] + 0.5, N1[0] * s + N2[n] + 0.5])
                # 计算并更新 raa[i]（加入距离的正则化值）
                raa[i] += myfun(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
    raa = raa / s**4  # 标准化正则化值
    return raa

@jit(nopython=True)
def deconvolution_fine(H, s, x_area, Sill_min, Range_min, L_sill, L_range, rate):
    """
    执行细化层的反卷积
    :param H: 滞后距离
    :param s: 缩放因子
    :param x_area: 参数区域
    :param Sill_min: 最小 Sill 值
    :param Range_min: 最小 Range 值
    :param L_sill: Sill 的最大数量
    :param L_range: Range 的最大数量
    :param rate: 调整步长
    :return: 最佳参数 x_best
    """

    # 计算 Fa0 向量
    Fa0 = myfun(x_area, np.arange(1, s * H + 1))
    Fa0_vector = Fa0[s - 1::s]  # 使用 Python 的切片语法选择每隔 s 个元素
    Dif_min = 10 ** 6  # 初始化最小差异值
    # 遍历 Sill 和 Range 的可能组合
    for i in range(1, L_sill + 1):
        for j in range(1, L_range + 1):
            # 更新 xp 参数
            xp = np.array([(Sill_min + rate * i) * x_area[0], (Range_min + rate * j) * x_area[1]])
            # 计算正则化值
            raa0 = regularization_fine(H, s, xp)
            raa = raa0[1:H + 1] - raa0[0]  # 计算差异值
            # 计算当前差异
            Dif = np.linalg.norm(raa - Fa0_vector)
            # 如果当前差异小于最小差异，更新最佳参数
            if Dif <= Dif_min:
                x_best = xp
                Dif_min = Dif
    return x_best

@jit(nopython=True)
def r_VV_kk(W, s0, s, xX, PSF):
    """
    计算 r_VV_kk
    :param W: 一个参数，影响 Assume_L1 大小
    :param s0: 子像素大小
    :param s: 缩放因子
    :param xX: 用于 myfun 的参数
    :param PSF: 点扩散函数
    :return: r1，结果
    """

    # 创建 Assume_L1 矩阵
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    # 初始化 r1
    r1 = np.zeros(((2 * W + 1) ** 2, (2 * W + 1) ** 2))
    # 遍历每个粗像素
    for i in range((2 * W + 1) ** 2):
        for j in range((2 * W + 1) ** 2):
            r_vV_kk = np.zeros(((2 * W + 1) * s0, (2 * W + 1) * s0))
            # 遍历每个粗像素的子像素
            for ii in range((2 * W + 1) * s0):
                for jj in range((2 * W + 1) * s0):
                    r_vv_kk = np.zeros(((2 * W + 1) * s0, (2 * W + 1) * s0))
                    # 遍历当前粗像素的子像素
                    for iii in range((2 * W + 1) * s0):
                        for jjj in range((2 * W + 1) * s0):
                            p1 = np.array([(M1[i] - W) * s * s0 + s * (ii + 1) - 1, (N1[i] - W) * s * s0 + s * (jj + 1) - 1])
                            p2 = np.array([(M1[j] - W) * s * s0 + s * (iii + 1) - 1, (N1[j]- W) * s * s0 + s * (jjj + 1) - 1])
                            r_vv_kk[iii, jjj] = myfun(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
                    # 计算 r_vV_kk
                    r_vV_kk[ii, jj] = np.sum(r_vv_kk * PSF)
            # 计算 r1
            r1[i, j] = np.sum(r_vV_kk * PSF)
    return r1

@jit(nopython=True)
def r_VU_kl(W1, W2, s0, s, xX, PSF1, PSF2):

    # 初始化Assume_L1和Assume_L2
    Assume_L1 = np.zeros((2 * W1 + 1, 2 * W1 + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    Assume_L2 = np.zeros((2 * W2 + 1, 2 * W2 + 1))
    M2, N2 = np.where(Assume_L2 == 0)
    r2 = np.zeros(((2 * W1 + 1) ** 2, (2 * W2 + 1) ** 2))
    # 对于每个粗像素 i 和 j
    for i in range((2 * W1 + 1) ** 2):
        for j in range((2 * W2 + 1) ** 2):
            TVv = np.zeros(((2 * W1 + 1) * s0, (2 * W1 + 1) * s0))
            # 对于粗像素 j 的局部窗口中的每个子像素
            for ii in range((2 * W1 + 1) * s0):
                for jj in range((2 * W1 + 1) * s0):
                    Tvv = np.zeros(((2 * W2 + 1) * s, (2 * W2 + 1) * s))
                    # 对于粗像素 i 的局部窗口中的每个子像素
                    for iii in range((2 * W2 + 1) * s):
                        for jjj in range((2 * W2 + 1) * s):
                            p1 = np.array([(M1[i] - W1) * s * s0 + (ii + 1) * s - 1, (N1[i] - W1) * s * s0 + (jj + 1) * s - 1])
                            p2 = np.array([(M2[j] - W2) * s + iii + 0.5, (N2[j] - W2) * s + jjj + 0.5])
                            # 计算Tvv的值
                            Tvv[iii, jjj] = myfun2(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
                    # 计算TVv的值
                    TVv[ii, jj] = np.sum(Tvv * PSF2)
            # 计算r2的值
            r2[i, j] = np.sum(TVv * PSF1)
    return r2

@jit(nopython=True)
def r_UU_ll(W, s, xX, PSF):
    """
    计算DSCK系数阵 r_UU_ll
    :param W: 一个参数，影响Assume_L1大小
    :param s: 缩放因子
    :param xX: 用于myfun的两个参数
    :param PSF: 点扩散函数（PSF），假设为一个矩阵
    :return: r4，计算出的相关性矩阵
    """

    # 创建Assume_L1
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    # 初始化r4
    r4 = np.zeros(((2 * W + 1) ** 2, (2 * W + 1) ** 2))
    # 遍历每对粗像素 i 和 j
    for i in range((2 * W + 1) ** 2):
        for j in range((2 * W + 1) ** 2):
            r_uU_ll = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))
            # 对于粗像素 j 的局部窗口中的每个子像素
            for ii in range((2 * W + 1) * s):
                for jj in range((2 * W + 1) * s):
                    r_uu_ll = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))
                    # 对于粗像素 i 的局部窗口中的每个子像素
                    for iii in range((2 * W + 1) * s):
                        for jjj in range((2 * W + 1) * s):
                            p1 = np.array([(M1[i] - W) * s + ii + 0.5, (N1[i] - W) * s + jj + 0.5])
                            p2 = np.array([(M1[j] - W) * s + iii + 0.5, (N1[j] - W) * s + jjj + 0.5])
                            # 计算r_uu_ll的值
                            r_uu_ll[iii, jjj] = myfun(xX, np.sqrt(np.sum((p1 - p2) ** 2)))
                    # 计算r_uU_ll的值
                    r_uU_ll[ii, jj] = np.sum(r_uu_ll * PSF)
            # 计算r4的值
            r4[i, j] = np.sum(r_uU_ll * PSF)
    return r4

@jit(nopython=True)
def r_UV_kk(p_vm, W, s0, s, xX, PSF):
    """
    计算DSCK右一系数 r_UV_kk
    :param p_vm: 用于计算距离的参考点
    :param W: 一个参数，影响Assume_L1大小
    :param s0: 缩放因子
    :param s: 缩放因子
    :param xX: 用于myfun的两个参数
    :param PSF: 点扩散函数（PSF），假设为一个矩阵
    :return: r5，计算出的相关性矩阵
    """

    # 创建Assume_L1
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    r5 = np.zeros(((2 * W + 1) ** 2, 1))
    # 遍历每对粗像素 i
    for i in range((2 * W + 1) ** 2):
        r_uV_kk = np.zeros(((2 * W + 1) * s0, (2 * W + 1) * s0))
        # 对于粗像素 i 的局部窗口中的每个子像素
        for iii in range((2 * W + 1) * s0):
            for jjj in range((2 * W + 1) * s0):
                p1 = np.array([(M1[i] - W) * s * s0 + s * (iii + 1) - 1, (N1[i] - W) * s * s0 + s * (jjj + 1) - 1])
                # 计算r_uV_kk的值
                r_uV_kk[iii, jjj] = myfun(xX, np.sqrt(np.sum((p_vm - p1) ** 2)))
        # 计算r5的值
        r5[i] = np.sum(r_uV_kk * PSF)
    return r5

@jit(nopython=True)
def r_UU_kl(p_vm, W, s, xX, PSF):
    """
    计算DSCK右一系数 r_UU_kl
    :param p_vm: 用于计算距离的参考点
    :param W: 一个参数，影响Assume_L1大小
    :param s: 缩放因子
    :param xX: 用于myfun2的两个参数
    :param PSF: 点扩散函数（PSF），假设为一个矩阵
    :return: r6，计算出的相关性矩阵
    """

    # 创建Assume_L1
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    r6 = np.zeros(((2 * W + 1) ** 2, 1))
    # 遍历每对粗像素 i
    for i in range((2 * W + 1) ** 2):
        r_uu_kl = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))
        # 对于粗像素 i 的局部窗口中的每个子像素
        for iii in range((2 * W + 1) * s):
            for jjj in range((2 * W + 1) * s):
                p1 = np.array([(M1[i] - W) * s + iii + 0.5, (N1[i] - W) * s + jjj + 0.5])
                # 计算r_uu_kl的值
                r_uu_kl[iii, jjj] = myfun2(xX, np.sqrt(np.sum((p_vm - p1) ** 2)))
        # 计算r6的值
        r6[i] = np.sum(r_uu_kl * PSF)
    return r6

def calculate_parameter(s0, s, W1, W2, xX1, xX2, xX3, PSF1, PSF2):

    r1 = r_VV_kk(W1, s0, s, xX1, PSF1)  # 求点扩散函数取的s,PSF1=6
    r2 = r_VU_kl(W1, W2, s0, s, xX3, PSF1, PSF2)  # PSF2为24
    r4 = r_UU_ll(W2, s, xX2, PSF2)
    system = DSCKSystemBuilder.build(r1, r2, r4)
    coefficient_count = (2 * W1 + 1) ** 2 + (2 * W2 + 1) ** 2 + 2
    yita = np.zeros((s0, s0, coefficient_count))  # 用来存储结果
    for i in range(s0):
        for j in range(s0):
            cordinate_vm = np.array([W1 * s * s0 + s *( i + 1) - 1, W1 * s * s0 + s * (j + 1) - 1])
            r5 = r_UV_kk(cordinate_vm, W1, s0, s, xX1, PSF1)
            r6 = r_UU_kl(cordinate_vm, W2, s, xX3, PSF2)
            Vector = DSCKSystemBuilder.rhs(r5, r6)
            yita[i, j, :] = KrigingSolver.solve(system.matrix, Vector).flatten()
    return yita

@jit(nopython=True)
def calculate_coordinate(s0, s, W1, W2, Coarse, Fine, yitaX):
    c, d = Coarse.shape
    Simulated_part = np.zeros((c - 2 * W1, d - 2 * W1))  # 去除粗影像扩展边界
    M1, N1 = np.where(Simulated_part == 0)  # 获取 Simulated_part 中等于 0 的位置
    numberM1 = len(M1)  # M1 中的元素个数
    M1 = M1 + W1  # 增加 W1 偏移
    N1 = N1 + W1  # 增加 W1 偏移
    P_vm = np.zeros((c * s0, d * s0))  # 初始化 P_vm 矩阵
    for k in range(numberM1):
        for i in range(s0):
            for j in range(s0):
                # 获取局部窗口数据
                Local_W1 = Coarse[M1[k] - W1:M1[k] + W1 + 1, N1[k] - W1:N1[k] + W1 + 1]
                fine_row = s0 * M1[k]
                fine_column = s0 * N1[k]
                Local_W2 = Fine[
                    fine_row - W2:fine_row + W2 + 1,
                    fine_column - W2:fine_column + W2 + 1,
                ]
                # 获取 yitaX 的特定切片
                co = yitaX[i, j, :-2].flatten() # 假   设 D3_D2 函数是处理 yitaX 的一部分
                data = np.hstack((Local_W1.flatten(), Local_W2.flatten()))
                # 更新 P_vm 中对应位置的值
                P_vm[M1[k] * s0 + i, N1[k] * s0 + j] = np.dot( co.astype(np.float32), data.astype(np.float32))
    return P_vm


def _fit_variogram_models(Coarse, Fine, Constant_min, Sill_min, Range_min,
                          L_sill, L_range, L_constant, rate, H, W1, PSF1,
                          s0=COARSE_SCALE, s=FINE_SCALE):
    """拟合并反卷积粗、细及交叉半变异模型。"""
    # x0 和 x1 的初始值
    x0 = np.array([100, 1])
    x1 = np.array([10, 100, 1])
    # 退化为低空间分辨率
    Fine_up = downsample_plane(Fine, s0, W1, PSF1)
    self_estimator = VariogramEstimator(
        empirical_kernel=semivariogram,
        residual_kernel=myfun_fit,
    )
    cross_estimator = CrossVariogramEstimator(
        empirical_kernel=semivariogram,
        cross_empirical_kernel=semivariogram_cross,
        residual_kernel=myfun2_fit,
    )
    coarse_fit = self_estimator.fit(
        Coarse, H, np.arange(s * s0, s * s0 * H + 1, s * s0), x0
    )
    xa1 = coarse_fit.parameters
    x_fine_best1 = deconvolution_coarse(H, s0, s, xa1, Sill_min, Range_min, L_sill, L_range, rate)
    fine_fit = self_estimator.fit(Fine, H, np.arange(s, s * H + 1, s), x0)
    xa2 = fine_fit.parameters
    x_fine_best2 = deconvolution_fine(H, s, xa2, Sill_min, Range_min, L_sill, L_range, rate)
    cross_fit = cross_estimator.fit_cross(
        Coarse, Fine_up, H, np.arange(s * s0, s * s0 * H + 1, s * s0), x1
    )
    xa3 = cross_fit.parameters
    x_fine_best3 = deconvolution_cross(H, s0, s, xa3, Constant_min, Sill_min, Range_min, L_sill, L_range, L_constant,
                                       rate)
    return x_fine_best1, x_fine_best2, x_fine_best3


def calculate_matrix(Coarse, Fine, Constant_min, Sill_min, Range_min, L_sill, L_range,
                            L_constant, rate, H, W1, W2, PSF1, PSF2,
                            s0=COARSE_SCALE, s=FINE_SCALE):
    x_fine_best1, x_fine_best2, x_fine_best3 = _fit_variogram_models(
        Coarse, Fine, Constant_min, Sill_min, Range_min, L_sill, L_range,
        L_constant, rate, H, W1, PSF1, s0, s
    )
    r1 = r_VV_kk(W1, s0, s, x_fine_best1, PSF1)  # 求点扩散函数取的s,PSF1=6
    r2 = r_VU_kl(W1, W2, s0, s, x_fine_best3, PSF1, PSF2)  # PSF2为24
    r3 = r2.T  # 转置r2
    r4 = r_UU_ll(W2, s, x_fine_best2, PSF2)
    # 修改后的Matrix构建
    Matrix1 = np.hstack((r1, r2, np.ones(((2 * W1 + 1) ** 2, 1)), np.zeros(((2 * W1 + 1) ** 2, 1))))
    Matrix2 = np.hstack((r3, r4, np.zeros(((2 * W2 + 1) ** 2, 1)), np.ones(((2 * W2 + 1) ** 2, 1))))
    Matrix3 = np.hstack((np.ones((1, (2 * W1 + 1) ** 2)), np.zeros((1, (2 * W2 + 1) ** 2)), np.zeros((1, 2))))
    Matrix4 = np.hstack((np.zeros((1, (2 * W1 + 1) ** 2)), np.ones((1, (2 * W2 + 1) ** 2)), np.zeros((1, 2))))
    Matrix = np.vstack((Matrix1, Matrix2, Matrix3, Matrix4))
    Vector = []
    for i in range(s0):
        for j in range(s0):
            cordinate_vm = np.array([W1 * s * s0 + s * (i + 1) - 1, W1 * s * s0 + s * (j + 1) - 1])
            r5 = r_UV_kk(cordinate_vm, W1, s0, s, x_fine_best1, PSF1)
            r6 = r_UU_kl(cordinate_vm, W2, s, x_fine_best3, PSF2)
            r0 = np.zeros((2, 1))
            r0[0] = 1
            r0[1] = 0
            vector = np.vstack((r5, r6, r0))
            Vector.append(vector)
    return Matrix, Vector


def DSCK_Regression_Sharpen(Coarse, Fine, Constant_min, Sill_min, Range_min, L_sill, L_range,
                            L_constant, rate, H, W1, W2, PSF1, PSF2,
                            s0=COARSE_SCALE, s=FINE_SCALE):
    # 扩展 Coarse 和 Fine
    Coarse_extend = extend_plane(Coarse, W1)
    Fine_extend = extend_plane(Fine, W2)
    x_fine_best1, x_fine_best2, x_fine_best3 = _fit_variogram_models(
        Coarse, Fine, Constant_min, Sill_min, Range_min, L_sill, L_range,
        L_constant, rate, H, W1, PSF1, s0, s
    )
    # 计算参数
    # matrix_left, matrix_right = calculate_matrix(s0, s, W1, W2, x_fine_best1, x_fine_best2, x_fine_best3, PSF1, PSF2)
    yita = calculate_parameter(s0, s, W1, W2, x_fine_best1, x_fine_best2, x_fine_best3, PSF1, PSF2)
    # 计算坐标
    P_vm = calculate_coordinate(s0, s, W1, W2, Coarse_extend, Fine_extend, yita)
    # 返回结果
    Z0 = P_vm[W1 * s0: -W1 * s0, W1 * s0: -W1 * s0]
    return Z0
