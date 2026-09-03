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
    extend_plane,
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
    W2, psf_sigma,
):
    """拟合并反卷积粗、细及交叉半变异模型(云感知 interpolate 模式)。

    与 dsck._fit_variogram_models 的 interpolate 分支一致,区别:
      - fine 自变异函数用 masked_semivariogram(跳过云像元对)
      - 交叉半方差用 masked_cross_semivariogram(跳过云像元对)
      - coarse 自变异函数无掩膜(coarse 无云)
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

    # ATPK 把 coarse 插值到 fine 尺度。
    atpk_psf = gaussian_psf(s0, W2, psf_sigma)
    Coarse_up = ATPK_Interpolate(
        Coarse, Sill_min, Range_min, L_sill, L_range, rate, H,
        W2, atpk_psf, s=s0, backend=backend,
    )

    coarse_emp = self_estimator.empirical(Coarse, H)
    fine_emp = fine_estimator.empirical(Fine, H)
    cross_emp = cross_estimator.empirical_cross(Coarse_up, Fine, H)

    x0_coarse = np.array([float(coarse_emp[-1]), float(np.median(coarse_dists))])
    x0_fine = np.array([float(max(fine_emp[-1], 1e-6)), float(np.median(fine_dists))])
    cross_sill0 = max(float(cross_emp[-1]), 1e-6)
    x1_cross = np.array([0.0, cross_sill0, float(np.median(fine_dists))])

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

    cross_fit = cross_estimator.fit_cross(Coarse_up, Fine, H, fine_dists, x1_cross)
    xa3 = cross_fit.parameters
    # interpolate 模式:交叉反卷积尺度 (1, s)
    if backend == "gpu":
        x_fine_best3 = deconvolution_cross_gpu(
            H, 1, s, xa3, Constant_min, Sill_min, Range_min,
            L_sill, L_range, L_constant, rate,
        )
    else:
        x_fine_best3 = deconvolution_cross(
            H, 1, s, xa3, Constant_min, Sill_min, Range_min, L_sill, L_range, L_constant, rate,
        )
    return x_fine_best1, x_fine_best2, x_fine_best3, Coarse_up


def CDSCK_Sharpen(
    Coarse, Fine, cloud_mask, Constant_min, Sill_min, Range_min,
    L_sill, L_range, L_constant, rate, H, W1, W2, PSF1, PSF2,
    s0, s, backend="cpu", psf_sigma=1.0, max_points=100, max_radius=50,
):
    """单波段云感知 DSCK 锐化。

    Parameters
    ----------
    Coarse, Fine : np.ndarray
        二维粗/细影像(Fine 与 cloud_mask 同形状)。
    cloud_mask : np.ndarray
        与 Fine 同形状,1=云,0=晴空。
    max_points : int
        每个预测点收集的最大非云点数(逐圈扩张窗口直到达到)。
    max_radius : int
        窗口扩张的最大半径(防无限扩张)。
    """
    if backend not in {"cpu", "gpu"}:
        raise ValueError("CDSCK backend 只能是 'cpu' 或 'gpu'。")
    if Fine.shape != cloud_mask.shape:
        raise ValueError("Fine 与 cloud_mask 形状必须一致。")

    # 扩展边界(coarse 用 W1,fine 用 W2)。
    Coarse_extend = extend_plane(Coarse, W1)
    Fine_extend = extend_plane(Fine, W2)
    mask_extend = extend_plane(cloud_mask.astype(np.int8), W2)

    x_fine_best1, x_fine_best2, x_fine_best3, Coarse_up = _cdsck_fit_variogram_models(
        Coarse, Fine, cloud_mask, Constant_min, Sill_min, Range_min,
        L_sill, L_range, L_constant, rate, H, W1, PSF1, s0, s, backend,
        W2, psf_sigma,
    )

    # 克里金系数矩阵(不依赖云,复用 dsck.calculate_parameter)。
    yita = calculate_parameter(
        s0, s, W1, W2, x_fine_best1, x_fine_best2, x_fine_best3, PSF1, PSF2, backend=backend,
    )

    # 逐点动态克里金预测。CPU 暂不支持(逐点求解过慢),需 GPU。
    if backend != "gpu":
        raise ValueError("c_dsck 暂只支持 GPU 后端(逐点动态克里金需 GPU 并行)。")
    from .gpu import cdsck_coordinate_gpu
    P_vm = cdsck_coordinate_gpu(
        s0, s, W1, W2, Coarse_extend, Fine_extend, mask_extend, yita,
        x_fine_best2, x_fine_best3, max_points, max_radius,
    )

    Z0 = P_vm[W1 * s0: -W1 * s0, W1 * s0: -W1 * s0]
    return Z0
