try:
    from . import spatial as GSF
except ImportError:  # 保留直接运行旧脚本时的兼容性
    import spatial as GSF
import numpy as np
from numba import jit
from scipy.optimize import least_squares
from .systems import ATPRKSystemBuilder, KrigingSolver

@jit(nopython=True)
def r_area_area2(h, s, xX):

    # Initialize the arrays
    Assume_L1 = np.zeros((h + 1, 1))  # Array of size (h+1,)
    M1, N1 = np.where(Assume_L1 == 0)  # Find indices where elements are zero (M1 and N1 are 1D arrays)
    Assume_L2 = np.zeros((s, s))  # A square grid of size s
    M2, N2 = np.where(Assume_L2 == 0)  # Find indices for the grid (M2 and N2 are 1D arrays)
    # Initialize raa array to store semivariogram results
    raa = np.zeros((h + 1, 1))
    # Compute the semivariogram for each lag distance
    for i in range(h + 1):
        raa[i, 0] = 0  # Reset the value at index i
        for m in range(s ** 2):
            for n in range(s ** 2):
                p1 = np.array([M1[i] * s + M2[m] + 0.5, N1[i] * s + N2[m] + 0.5])
                p2 = np.array([M1[0] * s + M2[n] + 0.5, N1[0] * s + N2[n] + 0.5])
                # Calculate the distance between points p1 and p2
                distance = np.sqrt(np.sum((p1 - p2) ** 2))
                # Add the result of the model function applied to the distance
                raa[i, 0] += GSF.myfun(xX, distance)
    # Normalize by s^4 (since the grid is s x s)
    raa = raa / s ** 4
    return raa

@jit(nopython=True)
def r_fine_coarse2(p_vm, W, s, xX, PSF):
    # 初始化 Assume_L1 和索引
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    rvV = np.zeros((len(M1), 1))  # 初始化 rvV
    for i in range((2 * W + 1) ** 2):  # 遍历局部窗口
        Tvv = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))  # 初始化 Tvv 矩阵
        for iii in range((2 * W + 1) * s):  # 遍历每个子像素
            for jjj in range((2 * W + 1) * s):
                # 计算当前粗像素位置的子像素位置
                p1 = np.array([(M1[i] - W) * s + iii + 0.5, (N1[i] - W) * s + jjj + 0.5])
                Tvv[iii, jjj] = GSF.myfun(xX, np.sqrt(np.sum((p_vm - p1) ** 2)))  # 使用 myfun2 计算 Tvv
        # 计算 rvV
        rvV[i, 0] = np.sum(Tvv * PSF)  # 计算 Tvv 和 PSF 的加权和
    return rvV

@jit(nopython=True)
def T_coarse_coarse2(W, s, xX, PSF):
    # 初始化 Assume_L1 和索引
    Assume_L1 = np.zeros((2 * W + 1, 2 * W + 1))
    M1, N1 = np.where(Assume_L1 == 0)
    # 初始化 TVV 矩阵
    TVV = np.zeros(((2 * W + 1) ** 2, (2 * W + 1) ** 2))
    for i in range((2 * W + 1) ** 2):  # 遍历所有粗像素 i
        for j in range((2 * W + 1) ** 2):  # 遍历所有粗像素 j
            TvV = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))  # 初始化 TvV 矩阵
            for ii in range((2 * W + 1) * s):  # 遍历当前粗像素 j 的所有子像素
                for jj in range((2 * W + 1) * s):
                    # 计算当前粗像素 i 和 j 的子像素位置
                    Tvv = np.zeros(((2 * W + 1) * s, (2 * W + 1) * s))  # 初始化 Tvv 矩阵
                    for iii in range((2 * W + 1) * s):  # 遍历当前粗像素 i 的所有子像素
                        for jjj in range((2 * W + 1) * s):
                            p1 = np.array([(M1[i] - W) * s + iii + 0.5, (N1[i] - W) * s + jjj + 0.5])
                            p2 = np.array([(M1[j] - W) * s + ii + 0.5, (N1[j] - W) * s + jj + 0.5])
                            Tvv[iii, jjj] = GSF.myfun(xX, np.sqrt(np.sum((p1 - p2) ** 2)))  # 计算 Tvv
                    # 更新 TvV 的值
                    TvV[ii, jj] = np.sum(Tvv * PSF)
            # 更新 TVV 的值
            TVV[i, j] = np.sum(TvV * PSF)
    return TVV

@jit(nopython=True)
def ATP_deconvolution(H, s, x_area, Sill_min, Range_min, L_sill, L_range, rate):

    # Apply the function to the coarse semivariogram
    Fa0 = GSF.myfun(x_area, np.arange(1, s * H + 1))
    Fa0_vector = Fa0[s - 1::s]  # Downsample by the scaling factor `s`
    Dif_min = 10 ** 6  # Initialize a very large difference value
    # Loop through all sill and range values to find the best match
    for i in range(1, L_sill + 1):  # sill loop
        for j in range(1, L_range + 1):  # range loop
            xp = np.array([(Sill_min + rate * i) * x_area[0], (Range_min + rate * j) * x_area[1]])
            raa0 = r_area_area2(H, s, xp)
            raa = raa0[1:H + 1, 0] - raa0[0, 0]  # Compute the difference of area values
            Dif = np.linalg.norm(raa - Fa0_vector)  # Compute the norm of the difference
            if Dif <= Dif_min:
                x_best = xp
                Dif_min = Dif
    return x_best

def calculate_parameter(s, W, xX, PSF):

    TVV = T_coarse_coarse2(W, s, xX, PSF)
    system = ATPRKSystemBuilder.build(TVV)
    yita = np.zeros((s, s, (2 * W + 1) ** 2 + 1))  # 用于存储结果
    RMSE = np.zeros((s, s))  # 用于存储 RMSE 结果
    for i in range(s):
        for j in range(s):
            cordinate_vm = np.array([W * s + i + 0.5, W * s + j + 0.5])
            rvV = r_fine_coarse2(cordinate_vm, W, s, xX, PSF)
            Vector = ATPRKSystemBuilder.rhs(rvV)
            yita[i, j, :] = KrigingSolver.solve(system.matrix, Vector).flatten()
            # 计算 2D RMSE
            yita_2D = yita[i, j, :].flatten()
            RMSE[i, j] = yita_2D.dot(Vector.flatten())  # 计算 RMSE
    return yita, RMSE

@jit(nopython=True)
def calculate_coordinate(s, W, S, yitaX, RMSE0):

    c, d = S.shape  # 获取 S 的尺寸
    Simulated_part = np.zeros((c - 2 * W, d - 2 * W))  # 计算扩展后的矩阵
    M1, N1 = np.where(Simulated_part == 0)  # 找到矩阵中为 0 的位置
    numberM1 = len(M1)  # 获取总数
    M1 = M1 + W  # 调整索引
    N1 = N1 + W  # 调整索引
    P_vm = np.zeros((c * s, d * s))  # 初始化结果矩阵 P_vm
    RMSE = np.zeros_like(P_vm)  # 初始化 RMSE 矩阵
    for k in range(numberM1):
        for i in range(s):
            for j in range(s):
                # 提取局部窗口
                Local_W = S[M1[k] - W: M1[k] + W + 1, N1[k] - W: N1[k] + W + 1]
                # 获取 yita 的变换值
                co = yitaX[i, j, :-1].flatten()  # 变换 yitaX，不包括最后一维
                # 更新 P_vm 矩阵
                data = Local_W.flatten()
                P_vm[M1[k] * s + i, N1[k] * s + j] = np.dot(co.astype(np.float32), data.astype(np.float32))
        # 更新 RMSE 矩阵
        RMSE[M1[k] * s: (M1[k] + 1) * s, N1[k] * s: (N1[k] + 1) * s] = RMSE0
    return P_vm, RMSE


def ATPRK_Sharpen(Coarse, PAN, Sill_min, Range_min, L_sill, L_range, rate, H, w, PSF):

    a1, b1 = Coarse.shape
    a2, b2 = PAN.shape
    s = int(a2 / a1)
    # Linear regression modeling
    PAN_upscaled = GSF.downsample_plane(PAN, s, w, PSF)
    PAN_upscaled_col = np.column_stack([PAN_upscaled.T.flatten(), np.ones(PAN_upscaled.size)])
    Coarse_col = Coarse.T.flatten().reshape(PAN_upscaled.size, 1)
    alpha = np.linalg.lstsq(PAN_upscaled_col, Coarse_col, rcond=None)[0]
    PAN_col = np.column_stack([PAN.T.flatten(), np.ones(PAN.size)])
    Z_R = PAN_col.dot(alpha).reshape(PAN.shape).T
    # Residual calculation
    Z_R_upscaled = GSF.downsample_plane(Z_R, s, w, PSF)
    RB = Coarse - Z_R_upscaled  # RB is the residual
    # ATPK for residuals, Deconvolution is achieved by trial-and-error
    W = w
    RB_extend = GSF.extend_plane(RB, W)
    x0 = [100, 1]  # Initial values for fitting
    rh = [GSF.semivariogram(RB, t) for t in range(1, H + 1)]
    # Fit the model using least squares
    result = least_squares(GSF.myfun_fit, x0, args=(np.arange(s, s * H + 1, s), rh))
    xa1 = result.x
    xp_best = ATP_deconvolution(H, s, xa1, Sill_min, Range_min, L_sill, L_range, rate)
    yita1, RMSE0 = calculate_parameter(s, W, xp_best, PSF)
    P_vm, RMSE = calculate_coordinate(s, W, RB_extend, yita1, RMSE0)
    Z_ATPK = P_vm[W * s: -W * s, W * s: -W * s]
    R = RMSE[W * s: -W * s, W * s: -W * s]
    Z = Z_R + Z_ATPK
    return R, Z
