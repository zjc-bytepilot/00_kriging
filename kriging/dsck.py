import numpy as np
from numba import jit
from .spatial import (
    PSF,
    cross_semivariogram as semivariogram_cross,
    downsample_plane,
    exponential_cross_variogram as myfun2,
    exponential_variogram as myfun,
    extend_plane,
    legacy_exponential_cross_variogram_residual as myfun2_fit,
    legacy_exponential_variogram_residual as myfun_fit,
    semivariogram,
)
from .support import (
    deconvolution_coarse,
    deconvolution_cross,
    deconvolution_fine,
    regularization_coarse,
    regularization_cross,
    regularization_fine,
)
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


def dowmsample_cube(cube, scale, window, psf):
    """Downsample every spectral plane using the canonical spatial primitive."""
    rows, columns, bands = cube.shape
    result = np.zeros((int(rows / scale), int(columns / scale), bands))
    for band in range(bands):
        result[:, :, band] = downsample_plane(cube[:, :, band], scale, window, psf)
    return result

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

def _build_kriging_system(coarse_variogram, cross_variogram, fine_variogram):
    """Adapt DSCK's legacy coefficient inputs to the shared system builder."""
    return DSCKSystemBuilder.build(coarse_variogram, cross_variogram, fine_variogram)


def calculate_parameter(s0, s, W1, W2, xX1, xX2, xX3, PSF1, PSF2):

    r1 = r_VV_kk(W1, s0, s, xX1, PSF1)  # 求点扩散函数取的s,PSF1=6
    r2 = r_VU_kl(W1, W2, s0, s, xX3, PSF1, PSF2)  # PSF2为24
    r4 = r_UU_ll(W2, s, xX2, PSF2)
    system = _build_kriging_system(r1, r2, r4)
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

    coarse_dists = np.arange(s * s0, s * s0 * H + 1, s * s0)
    fine_dists = np.arange(s, s * H + 1, s)
    coarse_emp = self_estimator.empirical(Coarse, H)
    fine_emp = self_estimator.empirical(Fine, H)
    cross_emp = cross_estimator.empirical_cross(Coarse, Fine_up, H)
    x0_coarse = np.array([float(coarse_emp[-1]), float(np.median(coarse_dists))])
    x0_fine = np.array([float(fine_emp[-1]), float(np.median(fine_dists))])
    x1_cross = np.array([0.0, float(cross_emp[-1]), float(np.median(coarse_dists))])
    coarse_fit = self_estimator.fit(Coarse, H, coarse_dists, x0_coarse)
    xa1 = coarse_fit.parameters
    x_fine_best1 = deconvolution_coarse(H, s0, s, xa1, Sill_min, Range_min, L_sill, L_range, rate)
    fine_fit = self_estimator.fit(Fine, H, fine_dists, x0_fine)
    xa2 = fine_fit.parameters
    x_fine_best2 = deconvolution_fine(H, s, xa2, Sill_min, Range_min, L_sill, L_range, rate)
    cross_fit = cross_estimator.fit_cross(Coarse, Fine_up, H, coarse_dists, x1_cross)
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
    r4 = r_UU_ll(W2, s, x_fine_best2, PSF2)
    Matrix = _build_kriging_system(r1, r2, r4).matrix
    Vector = []
    for i in range(s0):
        for j in range(s0):
            cordinate_vm = np.array([W1 * s * s0 + s * (i + 1) - 1, W1 * s * s0 + s * (j + 1) - 1])
            r5 = r_UV_kk(cordinate_vm, W1, s0, s, x_fine_best1, PSF1)
            r6 = r_UU_kl(cordinate_vm, W2, s, x_fine_best3, PSF2)
            vector = DSCKSystemBuilder.rhs(r5, r6)
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
