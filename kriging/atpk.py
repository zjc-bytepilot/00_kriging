"""纯面积到点克里金(ATPK),不含回归项。

ATPRK = 线性回归 + ATPK;本模块只做 ATPK 部分:对 coarse 拟合半方差、
反卷积到点支撑、建克里金方程、插值到 fine 分辨率。

用途有二:
  1. 供 DSCK 的 ``cross_mode="interpolate"`` 把 coarse 插值到 fine 尺度,
     以便在 fine 尺度下计算交叉半方差;
  2. 作为独立插值器直接使用。

与 :mod:`kriging.atprk` 的区别仅是去掉了回归项(``Z_R``)与残差路径,
直接对 coarse 本身做 ATPK。``calculate_parameter`` / ``calculate_coordinate``
与反卷积/GPU kernel 全部复用 atprk 既有实现,不重复造轮子。
"""

from __future__ import annotations

import numpy as np

from . import spatial as GSF
from .atprk import calculate_coordinate, calculate_parameter, r_fine_coarse2
from .support import atprk_deconvolution
from .systems import ATPRKSystemBuilder, KrigingSolver
from .variogram import VariogramEstimator


def _calculate_parameter_gpu(s, W, xX, PSF, t_coarse_coarse_gpu):
    """GPU 版 ATPK 克里金权重计算。

    与 :func:`kriging.atprk.calculate_parameter` 等价,仅把最耗时的
    coarse-coarse 系数矩阵 ``T_coarse_coarse2`` 换成 GPU kernel。
    ``r_fine_coarse2`` 仍走 CPU(规模小,非瓶颈)。
    """
    TVV = t_coarse_coarse_gpu(W, s, xX, PSF)
    system = ATPRKSystemBuilder.build(TVV)
    yita = np.zeros((s, s, (2 * W + 1) ** 2 + 1))
    RMSE = np.zeros((s, s))
    for i in range(s):
        for j in range(s):
            cordinate_vm = np.array([W * s + i + 0.5, W * s + j + 0.5])
            rvV = r_fine_coarse2(cordinate_vm, W, s, xX, PSF)
            Vector = ATPRKSystemBuilder.rhs(rvV)
            yita[i, j, :] = KrigingSolver.solve(system.matrix, Vector).flatten()
            RMSE[i, j] = yita[i, j, :].flatten().dot(Vector.flatten())
    return yita, RMSE


def ATPK_Interpolate(
    Coarse,
    Sill_min,
    Range_min,
    L_sill,
    L_range,
    rate,
    H,
    w,
    PSF,
    *,
    s=None,
    backend="cpu",
):
    """把 coarse 用纯 ATPK 插值到 fine 分辨率。

    Parameters
    ----------
    Coarse : np.ndarray
        二维粗分辨率影像。
    Sill_min, Range_min, L_sill, L_range, rate, H :
        变异函数反卷积搜索参数,语义同 :func:`kriging.atprk.ATPRK_Sharpen`。
    w : int
        ATPK 局部窗口半径(单侧)。
    PSF : np.ndarray
        与放大倍数 ``s`` 对应的点扩散函数。
    s : int, optional
        放大倍数。不传时由 ``PSF`` 尺度推导:``PSF.shape[0] // (2*w+1)``。
    backend : str
        ``"cpu"`` 或 ``"gpu"``。GPU 复用 atprk 的反卷积与坐标 kernel。
    """
    if backend not in {"cpu", "gpu"}:
        raise ValueError("ATPK backend 只能是 'cpu' 或 'gpu'。")
    if s is None:
        psf_side = PSF.shape[0]
        s = psf_side // (2 * w + 1)
        if s * (2 * w + 1) != psf_side:
            raise ValueError(
                f"PSF 边长 {psf_side} 必须是 (2*w+1)={2 * w + 1} 的整数倍,否则无法推导 s。"
            )

    W = w
    Coarse_extend = GSF.extend_plane(Coarse, W)

    estimator = VariogramEstimator()
    dists = np.arange(s, s * H + 1, s)
    emp = estimator.empirical(Coarse, H)
    x0 = np.array([float(emp[-1]), float(np.median(dists))])
    fit = estimator.fit(Coarse, max_lag=H, distances=dists, initial=x0)
    xa1 = fit.parameters

    if backend == "gpu":
        from .gpu import (
            atprk_coordinate_gpu,
            atprk_deconvolution_gpu,
            t_coarse_coarse_gpu,
        )

        xp_best = atprk_deconvolution_gpu(
            H, s, xa1, Sill_min, Range_min, L_sill, L_range, rate
        )
        yita1, RMSE0 = _calculate_parameter_gpu(s, W, xp_best, PSF, t_coarse_coarse_gpu)
        P_vm, _RMSE = atprk_coordinate_gpu(s, W, Coarse_extend, yita1, RMSE0)
    else:
        xp_best = atprk_deconvolution(
            H, s, xa1, Sill_min, Range_min, L_sill, L_range, rate
        )
        yita1, RMSE0 = calculate_parameter(s, W, xp_best, PSF)
        P_vm, _RMSE = calculate_coordinate(s, W, Coarse_extend, yita1, RMSE0)

    Z_ATPK = P_vm[W * s: -W * s, W * s: -W * s]
    return Z_ATPK
