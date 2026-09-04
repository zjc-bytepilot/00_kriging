"""云感知 DSCK(c_dsck)。

在 DSCK 的 interpolate 模式基础上引入云掩膜:对 fine 影像中有云遮挡的
区域,变异函数拟合时跳过云像元对,预测时局部窗口内动态重建克里金方程
(只用非云点)。

固定走 interpolate 模式:ATPK 把 coarse 插值到 fine 尺度,交叉半方差在
fine 尺度下计算。云掩膜为与 Fine 同形状的 0/1 数组,1 表示云。
"""

from __future__ import annotations

import numpy as np

from . import spatial as GSF
from .atpk import ATPK_Interpolate
from .dsck import calculate_parameter
from .spatial import (
    gaussian_psf,
    masked_cross_semivariogram,
    masked_semivariogram,
    semivariogram,
)
from .support import (
    deconvolution_coarse,
    deconvolution_cross,
    deconvolution_fine,
)
from .variogram import CrossVariogramEstimator, VariogramEstimator
from .spatial import legacy_exponential_variogram_residual as myfun_fit
from .spatial import legacy_exponential_cross_variogram_residual as myfun2_fit


def _cdsck_fit_variogram_models(
    Coarse, Fine, cloud_mask, Constant_min, Sill_min, Range_min,
    L_sill, L_range, L_constant, rate, H, W1, PSF1, s0, s, backend,
    W2, psf_sigma, cross_mode="interpolate",
):
    """拟合并反卷积粗、细及交叉半变异模型(云感知)。

    与 dsck._fit_variogram_models 一致,区别:
      - fine 自变异函数用 masked_semivariogram(跳过云像元对)
      - 交叉半方差用 masked_cross_semivariogram(跳过云像元对)
      - coarse 自变异函数无掩膜(coarse 无云)

    ``cross_mode`` 决定交叉经验变异函数的观测尺度:
      - ``"interpolate"``:ATPK 把 coarse 插值到 fine 尺度,交叉半方差
        在 fine 尺度下计算,反卷积尺度为 (1, s);
      - ``"degrade"``:把 Fine 退化到 coarse 尺度,交叉半方差在 coarse
        尺度下计算,反卷积尺度为 (s0, s)。
    两种模式的点尺度模型相同,均以 fine/s 为点尺度单位。
    """
    self_estimator = VariogramEstimator(
        empirical_kernel=semivariogram,
        residual_kernel=myfun_fit,
    )
    # fine 自变异函数:掩膜版。用闭包固定 mask。
    fine_mask = cloud_mask.astype(np.int8)

    def _fine_emp_kernel(plane, lag):
        return masked_semivariogram(plane, fine_mask, lag)

    fine_estimator = VariogramEstimator(
        empirical_kernel=_fine_emp_kernel,
        residual_kernel=myfun_fit,
    )
    cross_estimator = CrossVariogramEstimator(
        empirical_kernel=semivariogram,
        cross_empirical_kernel=lambda a, b, lag: masked_cross_semivariogram(a, b, fine_mask, lag),
        residual_kernel=myfun2_fit,
    )

    coarse_dists = np.arange(s * s0, s * s0 * H + 1, s * s0)
    fine_dists = np.arange(s, s * H + 1, s)

    # 交叉半方差配对:interpolate 插值 coarse,degrade 退化 fine。
    if cross_mode == "interpolate":
        # ATPK 把 coarse 插值到 fine 尺度。
        atpk_psf = gaussian_psf(s0, W2, psf_sigma)
        Coarse_up = ATPK_Interpolate(
            Coarse, Sill_min, Range_min, L_sill, L_range, rate, H,
            W2, atpk_psf, s=s0, backend=backend,
        )
        cross_first, cross_second = Coarse_up, Fine
        cross_dists = fine_dists
    else:
        Fine_up = GSF.downsample_plane(Fine, s0, W1, PSF1)
        cross_first, cross_second = Coarse, Fine_up
        cross_dists = coarse_dists

    coarse_emp = self_estimator.empirical(Coarse, H)
    fine_emp = fine_estimator.empirical(Fine, H)
    cross_emp = cross_estimator.empirical_cross(cross_first, cross_second, H)

    x0_coarse = np.array([float(coarse_emp[-1]), float(np.median(coarse_dists))])
    x0_fine = np.array([float(max(fine_emp[-1], 1e-6)), float(np.median(fine_dists))])
    cross_sill0 = max(float(cross_emp[-1]), 1e-6)
    x1_cross = np.array([0.0, cross_sill0, float(np.median(cross_dists))])

    coarse_fit = self_estimator.fit(Coarse, H, coarse_dists, x0_coarse)
    xa1 = coarse_fit.parameters
    if backend == "gpu":
        from .gpu import (
            deconvolution_coarse_gpu,
            deconvolution_cross_gpu,
            deconvolution_fine_gpu,
        )
        x_fine_best1 = deconvolution_coarse_gpu(
            H, s0, s, xa1, Sill_min, Range_min, L_sill, L_range, rate,
        )
    else:
        x_fine_best1 = deconvolution_coarse(H, s0, s, xa1, Sill_min, Range_min, L_sill, L_range, rate)

    fine_fit = fine_estimator.fit(Fine, H, fine_dists, x0_fine)
    xa2 = fine_fit.parameters
    if backend == "gpu":
        x_fine_best2 = deconvolution_fine_gpu(H, s, xa2, Sill_min, Range_min, L_sill, L_range, rate)
    else:
        x_fine_best2 = deconvolution_fine(H, s, xa2, Sill_min, Range_min, L_sill, L_range, rate)

    cross_fit = cross_estimator.fit_cross(cross_first, cross_second, H, cross_dists, x1_cross)
    xa3 = cross_fit.parameters
    # 交叉反卷积尺度跟随交叉经验变异函数的观测尺度。
    cross_coarse_scale = 1 if cross_mode == "interpolate" else s0
    if backend == "gpu":
        x_fine_best3 = deconvolution_cross_gpu(
            H, cross_coarse_scale, s, xa3, Constant_min, Sill_min, Range_min,
            L_sill, L_range, L_constant, rate,
        )
    else:
        x_fine_best3 = deconvolution_cross(
            H, cross_coarse_scale, s, xa3, Constant_min, Sill_min, Range_min, L_sill, L_range, L_constant, rate,
        )
    return x_fine_best1, x_fine_best2, x_fine_best3, Coarse_up


def CDSCK_Sharpen(
    Coarse, Fine, cloud_mask, Constant_min, Sill_min, Range_min,
    L_sill, L_range, L_constant, rate, H, W1, W2, PSF1, PSF2,
    s0, s, backend="cpu", psf_sigma=1.0, cross_mode="interpolate",
    matrix_s0=None, matrix_s=None,
    max_points=100, max_radius=50, batch_size=512,
):
    """单波段云感知 DSCK 锐化。

    Parameters
    ----------
    Coarse, Fine : np.ndarray
        二维粗/细影像(Fine 与 cloud_mask 同形状)。
    cloud_mask : np.ndarray
        与 Fine 同形状,1=云,0=晴空。
    cross_mode : str
        交叉半方差配对策略:``"interpolate"`` 或 ``"degrade"``。
    matrix_s0, matrix_s : int, optional
        克里金矩阵阶段的 s0/s,为 None 时回退到经验阶段取值;
        ``matrix_s0`` 必须等于 ``s0``(输出网格约束)。
    max_points : int
        每个 coarse 局部窗口的有效 fine 点容量上限。
    max_radius : int
        自适应 fine 窗口的最大半径(初始为 W2,有效点不超过 50 时扩张)。
    """
    if backend not in {"cpu", "gpu"}:
        raise ValueError("CDSCK backend 只能是 'cpu' 或 'gpu'。")
    if Fine.shape != cloud_mask.shape:
        raise ValueError("Fine 与 cloud_mask 形状必须一致。")
    if matrix_s0 is None:
        matrix_s0 = s0
    if matrix_s is None:
        matrix_s = s
    if matrix_s0 != s0:
        raise ValueError("matrix_s0 必须等于 s0:输出网格由 coarse×s0 决定。")

    x_fine_best1, x_fine_best2, x_fine_best3, Coarse_up = _cdsck_fit_variogram_models(
        Coarse, Fine, cloud_mask, Constant_min, Sill_min, Range_min,
        L_sill, L_range, L_constant, rate, H, W1, PSF1, s0, s, backend,
        W2, psf_sigma, cross_mode,
    )

    # 克里金系数矩阵(不依赖云,复用 dsck.calculate_parameter)。
    yita = calculate_parameter(
        matrix_s0, matrix_s, W1, W2, x_fine_best1, x_fine_best2, x_fine_best3, PSF1, PSF2, backend=backend,
    )

    # 按 coarse 局部窗口共享候选点,对每个 fine 子像元动态求解。CPU 暂不支持。
    if backend != "gpu":
        raise ValueError("c_dsck 暂只支持 GPU 后端(逐点动态克里金需 GPU 并行)。")
    from .cdsck_batch import cdsck_coordinate_batched_gpu
    P_vm = cdsck_coordinate_batched_gpu(
        s0, W1, W2,
        np.ascontiguousarray(Coarse),
        np.ascontiguousarray(Fine),
        np.ascontiguousarray(cloud_mask.astype(np.int8)),
        yita,
        x_fine_best1, x_fine_best2, x_fine_best3,
        max_points, max_radius, batch_size,
    )

    return P_vm
