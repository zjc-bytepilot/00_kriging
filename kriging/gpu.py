"""CUDA acceleration kernels for kriging while preserving CPU algorithms."""

from __future__ import annotations

import math

import numpy as np
from numba import cuda

from .spatial import exponential_variogram


def is_available() -> bool:
    """Return whether this process can launch CUDA kernels."""
    return cuda.is_available()


def _require_cuda() -> None:
    if not is_available():
        raise RuntimeError("GPU 后端需要可用的 CUDA 设备。")


@cuda.jit
def _atprk_regularization_grid_kernel(
    regularized,
    h,
    scale,
    area_sill,
    area_range,
    sill_min,
    range_min,
    range_steps,
    rate,
):
    candidate_index, lag = cuda.grid(2)
    candidate_count = regularized.shape[0]
    if candidate_index >= candidate_count or lag > h:
        return

    sill_step = candidate_index // range_steps + 1
    range_step = candidate_index % range_steps + 1
    sill = (sill_min + rate * sill_step) * area_sill
    variogram_range = (range_min + rate * range_step) * area_range
    total = 0.0
    for first_row in range(scale):
        for first_column in range(scale):
            for second_row in range(scale):
                for second_column in range(scale):
                    row_delta = lag * scale + first_row - second_row
                    column_delta = first_column - second_column
                    distance = math.sqrt(row_delta * row_delta + column_delta * column_delta)
                    total += sill * (1.0 - math.exp(-distance / variogram_range))
    regularized[candidate_index, lag] = total / (scale ** 4)


def atprk_deconvolution_gpu(
    h: int,
    scale: int,
    area_parameters: np.ndarray,
    sill_min: float,
    range_min: float,
    sill_steps: int,
    range_steps: int,
    rate: float,
) -> np.ndarray:
    """Evaluate ATPRK's full candidate grid on CUDA and return the best pair."""
    _require_cuda()
    parameters = np.asarray(area_parameters, dtype=np.float64)
    if parameters.shape != (2,):
        raise ValueError("area_parameters 必须包含 sill 和 range 两个参数。")
    if min(h, scale, sill_steps, range_steps) <= 0:
        raise ValueError("h、scale、sill_steps 和 range_steps 必须为正整数。")
    if rate <= 0:
        raise ValueError("rate 必须大于 0。")

    candidate_count = sill_steps * range_steps
    regularized = cuda.device_array((candidate_count, h + 1), dtype=np.float64)
    threads = (16, 16)
    blocks = (
        (candidate_count + threads[0] - 1) // threads[0],
        (h + 1 + threads[1] - 1) // threads[1],
    )
    _atprk_regularization_grid_kernel[blocks, threads](
        regularized,
        h,
        scale,
        parameters[0],
        parameters[1],
        sill_min,
        range_min,
        range_steps,
        rate,
    )
    regularized_host = regularized.copy_to_host()
    area_samples = exponential_variogram(
        parameters,
        np.arange(scale, scale * h + 1, scale),
    )
    differences = np.linalg.norm(
        regularized_host[:, 1:] - regularized_host[:, :1] - area_samples,
        axis=1,
    )
    best_index = np.flatnonzero(differences <= np.min(differences))[-1]
    sill_step = best_index // range_steps + 1
    range_step = best_index % range_steps + 1
    return np.asarray(
        [
            (sill_min + rate * sill_step) * parameters[0],
            (range_min + rate * range_step) * parameters[1],
        ],
        dtype=np.float64,
    )


@cuda.jit
def _coarse_regularization_grid_kernel(
    regularized, h, coarse_scale, fine_scale, area_sill, area_range,
    sill_min, range_min, range_steps, rate,
):
    candidate_index, lag = cuda.grid(2)
    if candidate_index >= regularized.shape[0] or lag > h:
        return
    sill_step = candidate_index // range_steps + 1
    range_step = candidate_index % range_steps + 1
    sill = (sill_min + rate * sill_step) * area_sill
    variogram_range = (range_min + rate * range_step) * area_range
    total = 0.0
    for first_row in range(coarse_scale):
        for first_column in range(coarse_scale):
            for second_row in range(coarse_scale):
                for second_column in range(coarse_scale):
                    row_delta = lag * fine_scale * coarse_scale + fine_scale * (first_row - second_row)
                    column_delta = fine_scale * (first_column - second_column)
                    distance = math.sqrt(row_delta * row_delta + column_delta * column_delta)
                    total += sill * (1.0 - math.exp(-distance / variogram_range))
    regularized[candidate_index, lag] = total / (coarse_scale ** 4)


@cuda.jit
def _fine_regularization_grid_kernel(
    regularized, h, fine_scale, area_sill, area_range,
    sill_min, range_min, range_steps, rate,
):
    candidate_index, lag = cuda.grid(2)
    if candidate_index >= regularized.shape[0] or lag > h:
        return
    sill_step = candidate_index // range_steps + 1
    range_step = candidate_index % range_steps + 1
    sill = (sill_min + rate * sill_step) * area_sill
    variogram_range = (range_min + rate * range_step) * area_range
    total = 0.0
    for first_row in range(fine_scale):
        for first_column in range(fine_scale):
            for second_row in range(fine_scale):
                for second_column in range(fine_scale):
                    row_delta = lag * fine_scale + first_row - second_row
                    column_delta = first_column - second_column
                    distance = math.sqrt(row_delta * row_delta + column_delta * column_delta)
                    total += sill * (1.0 - math.exp(-distance / variogram_range))
    regularized[candidate_index, lag] = total / (fine_scale ** 4)


@cuda.jit
def _cross_regularization_grid_kernel(
    regularized, h, coarse_scale, fine_scale, area_constant, area_sill, area_range,
    constant_min, sill_min, range_min, sill_steps, range_steps, rate,
):
    candidate_index, lag = cuda.grid(2)
    if candidate_index >= regularized.shape[0] or lag > h:
        return
    per_constant = sill_steps * range_steps
    constant_step = candidate_index // per_constant + 1
    remaining = candidate_index % per_constant
    sill_step = remaining // range_steps + 1
    range_step = remaining % range_steps + 1
    constant = (constant_min + rate * constant_step) * area_constant
    sill = (sill_min + rate * sill_step) * area_sill
    variogram_range = (range_min + rate * range_step) * area_range
    total = 0.0
    fine_size = fine_scale * coarse_scale
    for coarse_row in range(coarse_scale):
        for coarse_column in range(coarse_scale):
            for fine_row in range(fine_size):
                for fine_column in range(fine_size):
                    row_delta = lag * fine_size + fine_row - fine_scale * coarse_row - 0.5
                    column_delta = fine_column - fine_scale * coarse_column - 0.5
                    distance = math.sqrt(row_delta * row_delta + column_delta * column_delta)
                    total += constant + sill * (1.0 - math.exp(-distance / variogram_range))
    regularized[candidate_index, lag] = total / (coarse_scale * fine_scale * coarse_scale) ** 2


def _launch_regularization(kernel, candidate_count: int, h: int, *arguments) -> np.ndarray:
    regularized = cuda.device_array((candidate_count, h + 1), dtype=np.float64)
    threads = (16, 16)
    blocks = (
        (candidate_count + threads[0] - 1) // threads[0],
        (h + 1 + threads[1] - 1) // threads[1],
    )
    kernel[blocks, threads](regularized, h, *arguments)
    return regularized.copy_to_host()


def _select_two_parameter_candidate(
    regularized: np.ndarray,
    area_parameters: np.ndarray,
    sample_step: int,
    sill_min: float,
    range_min: float,
    range_steps: int,
    rate: float,
) -> np.ndarray:
    h = regularized.shape[1] - 1
    samples = exponential_variogram(
        area_parameters, np.arange(sample_step, sample_step * h + 1, sample_step),
    )
    differences = np.linalg.norm(regularized[:, 1:] - regularized[:, :1] - samples, axis=1)
    best_index = np.flatnonzero(differences <= np.min(differences))[-1]
    sill_step = best_index // range_steps + 1
    range_step = best_index % range_steps + 1
    return np.asarray(
        [
            (sill_min + rate * sill_step) * area_parameters[0],
            (range_min + rate * range_step) * area_parameters[1],
        ],
        dtype=np.float64,
    )


def deconvolution_coarse_gpu(
    h: int, coarse_scale: int, fine_scale: int, area_parameters: np.ndarray,
    sill_min: float, range_min: float, sill_steps: int, range_steps: int, rate: float,
) -> np.ndarray:
    """CUDA candidate search matching ``support.deconvolution_coarse``."""
    _require_cuda()
    parameters = np.asarray(area_parameters, dtype=np.float64)
    candidate_count = sill_steps * range_steps
    regularized = _launch_regularization(
        _coarse_regularization_grid_kernel, candidate_count, h,
        coarse_scale, fine_scale, parameters[0], parameters[1],
        sill_min, range_min, range_steps, rate,
    )
    return _select_two_parameter_candidate(
        regularized, parameters, fine_scale * coarse_scale,
        sill_min, range_min, range_steps, rate,
    )


def deconvolution_fine_gpu(
    h: int, fine_scale: int, area_parameters: np.ndarray,
    sill_min: float, range_min: float, sill_steps: int, range_steps: int, rate: float,
) -> np.ndarray:
    """CUDA candidate search matching ``support.deconvolution_fine``."""
    _require_cuda()
    parameters = np.asarray(area_parameters, dtype=np.float64)
    candidate_count = sill_steps * range_steps
    regularized = _launch_regularization(
        _fine_regularization_grid_kernel, candidate_count, h,
        fine_scale, parameters[0], parameters[1], sill_min, range_min, range_steps, rate,
    )
    return _select_two_parameter_candidate(
        regularized, parameters, fine_scale, sill_min, range_min, range_steps, rate,
    )


def deconvolution_cross_gpu(
    h: int, coarse_scale: int, fine_scale: int, area_parameters: np.ndarray,
    constant_min: float, sill_min: float, range_min: float,
    sill_steps: int, range_steps: int, constant_steps: int, rate: float,
) -> np.ndarray:
    """CUDA candidate search matching ``support.deconvolution_cross``."""
    _require_cuda()
    parameters = np.asarray(area_parameters, dtype=np.float64)
    candidate_count = constant_steps * sill_steps * range_steps
    regularized = _launch_regularization(
        _cross_regularization_grid_kernel, candidate_count, h,
        coarse_scale, fine_scale, parameters[0], parameters[1], parameters[2],
        constant_min, sill_min, range_min, sill_steps, range_steps, rate,
    )
    sample_step = fine_scale * coarse_scale
    distances = np.arange(sample_step, sample_step * h + 1, sample_step)
    samples = parameters[0] + parameters[1] * (1.0 - np.exp(-distances / parameters[2]))
    differences = np.linalg.norm(regularized[:, 1:] - regularized[:, :1] - samples, axis=1)
    best_index = np.flatnonzero(differences <= np.min(differences))[-1]
    per_constant = sill_steps * range_steps
    constant_step = best_index // per_constant + 1
    remaining = best_index % per_constant
    sill_step = remaining // range_steps + 1
    range_step = remaining % range_steps + 1
    return np.asarray(
        [
            (constant_min + rate * constant_step) * parameters[0],
            (sill_min + rate * sill_step) * parameters[1],
            (range_min + rate * range_step) * parameters[2],
        ],
        dtype=np.float64,
    )


@cuda.jit
def _atprk_coordinate_kernel(prediction, uncertainty, source, weights, rmse, scale, window):
    row, column = cuda.grid(2)
    if row >= prediction.shape[0] or column >= prediction.shape[1]:
        return
    coarse_row = row // scale
    coarse_column = column // scale
    if coarse_row < window or coarse_row >= source.shape[0] - window:
        return
    if coarse_column < window or coarse_column >= source.shape[1] - window:
        return
    sub_row = row % scale
    sub_column = column % scale
    coefficient = 0
    value = 0.0
    for local_row in range(-window, window + 1):
        for local_column in range(-window, window + 1):
            value += weights[sub_row, sub_column, coefficient] * source[
                coarse_row + local_row, coarse_column + local_column
            ]
            coefficient += 1
    prediction[row, column] = value
    uncertainty[row, column] = rmse[sub_row, sub_column]


def atprk_coordinate_gpu(
    scale: int, window: int, source: np.ndarray, weights: np.ndarray, rmse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct ATPRK prediction and uncertainty on CUDA."""
    _require_cuda()
    source_device = cuda.to_device(np.asarray(source, dtype=np.float64))
    weights_device = cuda.to_device(np.asarray(weights, dtype=np.float64))
    rmse_device = cuda.to_device(np.asarray(rmse, dtype=np.float64))
    shape = (source.shape[0] * scale, source.shape[1] * scale)
    prediction = cuda.device_array(shape, dtype=np.float64)
    uncertainty = cuda.device_array(shape, dtype=np.float64)
    threads = (16, 16)
    blocks = ((shape[0] + 15) // 16, (shape[1] + 15) // 16)
    _atprk_coordinate_kernel[blocks, threads](
        prediction, uncertainty, source_device, weights_device, rmse_device, scale, window,
    )
    return prediction.copy_to_host(), uncertainty.copy_to_host()


@cuda.jit
def _dsck_coordinate_kernel(prediction, coarse, fine, weights, coarse_scale, coarse_window, fine_window):
    row, column = cuda.grid(2)
    if row >= prediction.shape[0] or column >= prediction.shape[1]:
        return
    coarse_row = row // coarse_scale
    coarse_column = column // coarse_scale
    if coarse_row < coarse_window or coarse_row >= coarse.shape[0] - coarse_window:
        return
    if coarse_column < coarse_window or coarse_column >= coarse.shape[1] - coarse_window:
        return
    sub_row = row % coarse_scale
    sub_column = column % coarse_scale
    coefficient = 0
    value = 0.0
    for local_row in range(-coarse_window, coarse_window + 1):
        for local_column in range(-coarse_window, coarse_window + 1):
            value += weights[sub_row, sub_column, coefficient] * coarse[
                coarse_row + local_row, coarse_column + local_column
            ]
            coefficient += 1
    fine_row = coarse_scale * coarse_row
    fine_column = coarse_scale * coarse_column
    for local_row in range(-fine_window, fine_window + 1):
        for local_column in range(-fine_window, fine_window + 1):
            value += weights[sub_row, sub_column, coefficient] * fine[
                fine_row + local_row, fine_column + local_column
            ]
            coefficient += 1
    prediction[row, column] = value


def dsck_coordinate_gpu(
    coarse_scale: int, coarse_window: int, fine_window: int,
    coarse: np.ndarray, fine: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Reconstruct DSCK prediction on CUDA using the CPU-computed weights."""
    _require_cuda()
    coarse_device = cuda.to_device(np.asarray(coarse, dtype=np.float64))
    fine_device = cuda.to_device(np.asarray(fine, dtype=np.float64))
    weights_device = cuda.to_device(np.asarray(weights, dtype=np.float64))
    shape = (coarse.shape[0] * coarse_scale, coarse.shape[1] * coarse_scale)
    prediction = cuda.to_device(np.zeros(shape, dtype=np.float64))
    threads = (16, 16)
    blocks = ((shape[0] + 15) // 16, (shape[1] + 15) // 16)
    _dsck_coordinate_kernel[blocks, threads](
        prediction, coarse_device, fine_device, weights_device,
        coarse_scale, coarse_window, fine_window,
    )
    return prediction.copy_to_host()


@cuda.jit
def _t_coarse_coarse_kernel(tvv, window, scale, sill, rng, psf):
    """Compute one element of the ATPK coarse-coarse coefficient matrix TVV.

    Mirrors ``kriging.atprk.T_coarse_coarse2``: each thread computes TVV[i,j]
    by double PSF-weighted aggregation over sub-pixels of coarse cells i and j.
    ``sill``/``rng`` are the two variogram parameters (exponential model).
    """
    idx = cuda.grid(1)
    n = (2 * window + 1) ** 2
    if idx >= n * n:
        return
    i = idx // n
    j = idx % n
    side = (2 * window + 1) * scale
    # coarse cell index (M1, N1) within the (2W+1)x(2W+1) window.
    mi = i // (2 * window + 1) - window
    ni = i % (2 * window + 1) - window
    mj = j // (2 * window + 1) - window
    nj = j % (2 * window + 1) - window

    tvv_ij = 0.0
    # Outer aggregation: sub-pixel (ii,jj) of coarse cell j.
    for ii in range(side):
        for jj in range(side):
            inner = 0.0  # inner Tvv block
            for iii in range(side):
                for jjj in range(side):
                    p1x = mi * scale + iii + 0.5
                    p1y = ni * scale + jjj + 0.5
                    p2x = mj * scale + ii + 0.5
                    p2y = nj * scale + jj + 0.5
                    dx = p1x - p2x
                    dy = p1y - p2y
                    dist = math.sqrt(dx * dx + dy * dy)
                    inner += sill * (1.0 - math.exp(-dist / rng)) * psf[iii, jjj]
            tvv_ij += inner * psf[ii, jj]
    tvv[i, j] = tvv_ij


def t_coarse_coarse_gpu(
    window: int, scale: int, parameters: np.ndarray, psf: np.ndarray,
) -> np.ndarray:
    """GPU equivalent of ``kriging.atprk.T_coarse_coarse2``."""
    _require_cuda()
    n = (2 * window + 1) ** 2
    tvv = cuda.device_array((n, n), dtype=np.float64)
    psf_device = cuda.to_device(np.asarray(psf, dtype=np.float64))
    sill = float(parameters[0])
    rng = float(parameters[1])
    threads = 128
    blocks = (n * n + threads - 1) // threads
    _t_coarse_coarse_kernel[blocks, threads](
        tvv, window, scale, sill, rng, psf_device,
    )
    return tvv.copy_to_host()


@cuda.jit
def _r_uu_ll_kernel(r4, window, scale, sill, rng, psf):
    """Compute one element of the DSCK fine-fine coefficient matrix r_UU_ll.

    Mirrors ``kriging.dsck.r_UU_ll``: each thread computes r4[i,j] by double
    PSF-weighted aggregation. ``sill``/``rng`` are the exponential variogram
    parameters.
    """
    idx = cuda.grid(1)
    n = (2 * window + 1) ** 2
    if idx >= n * n:
        return
    i = idx // n
    j = idx % n
    side = (2 * window + 1) * scale
    mi = i // (2 * window + 1) - window
    ni = i % (2 * window + 1) - window
    mj = j // (2 * window + 1) - window
    nj = j % (2 * window + 1) - window

    r4_ij = 0.0
    # p1 用粗像元 i + 子像素 (ii,jj);p2 用粗像元 j + 子像素 (iii,jjj)
    for ii in range(side):
        for jj in range(side):
            inner = 0.0
            for iii in range(side):
                for jjj in range(side):
                    p1x = mi * scale + ii + 0.5
                    p1y = ni * scale + jj + 0.5
                    p2x = mj * scale + iii + 0.5
                    p2y = nj * scale + jjj + 0.5
                    dx = p1x - p2x
                    dy = p1y - p2y
                    dist = math.sqrt(dx * dx + dy * dy)
                    inner += sill * (1.0 - math.exp(-dist / rng)) * psf[iii, jjj]
            r4_ij += inner * psf[ii, jj]
    r4[i, j] = r4_ij


def r_uu_ll_gpu(
    window: int, scale: int, parameters: np.ndarray, psf: np.ndarray,
) -> np.ndarray:
    """GPU equivalent of ``kriging.dsck.r_UU_ll``."""
    _require_cuda()
    n = (2 * window + 1) ** 2
    r4 = cuda.device_array((n, n), dtype=np.float64)
    psf_device = cuda.to_device(np.asarray(psf, dtype=np.float64))
    sill = float(parameters[0])
    rng = float(parameters[1])
    threads = 128
    blocks = (n * n + threads - 1) // threads
    _r_uu_ll_kernel[blocks, threads](
        r4, window, scale, sill, rng, psf_device,
    )
    return r4.copy_to_host()


__all__ = [
    "atprk_coordinate_gpu",
    "atprk_deconvolution_gpu",
    "deconvolution_coarse_gpu",
    "deconvolution_cross_gpu",
    "deconvolution_fine_gpu",
    "dsck_coordinate_gpu",
    "is_available",
    "r_uu_ll_gpu",
    "t_coarse_coarse_gpu",
]
