"""Batched CUDA reconstruction for cloud-affected DSCK windows."""

from __future__ import annotations

import math

import cupy as cp
import numpy as np
from numba import cuda
from scipy.ndimage import maximum_filter

from .gpu import dsck_coordinate_gpu


def _adaptive_window_sizes(
    mask: np.ndarray,
    coarse_scale: int,
    coarse_window: int,
    initial_window: int,
    target_points: int,
    max_window: int,
) -> np.ndarray:
    """Choose one MATLAB-compatible fine window for every coarse cell."""
    coarse_rows = mask.shape[0] // coarse_scale
    coarse_columns = mask.shape[1] // coarse_scale
    windows = np.full((coarse_rows, coarse_columns), initial_window, dtype=np.int32)
    for coarse_row in range(coarse_window, coarse_rows - coarse_window):
        for coarse_column in range(coarse_window, coarse_columns - coarse_window):
            window = initial_window
            while True:
                center_row = coarse_row * coarse_scale
                center_column = coarse_column * coarse_scale
                row_start = max(0, center_row - window)
                row_end = min(mask.shape[0], center_row + window + 1)
                column_start = max(0, center_column - window)
                column_end = min(mask.shape[1], center_column + window + 1)
                valid_points = int(np.count_nonzero(mask[row_start:row_end, column_start:column_end] == 0))
                if valid_points > target_points or window >= max_window:
                    windows[coarse_row, coarse_column] = window
                    break
                window += 1
    return windows


@cuda.jit
def _gather_candidates(
    rows, columns, fine, mask, window_sizes, coarse_scale, max_points, xs, ys, values, counts
):
    index = cuda.grid(1)
    if index >= rows.size:
        return
    row = rows[index]
    column = columns[index]
    coarse_row = row // coarse_scale
    coarse_column = column // coarse_scale
    center_row = coarse_row * coarse_scale
    center_column = coarse_column * coarse_scale
    window = window_sizes[coarse_row, coarse_column]
    count = 0
    for dr in range(-window, window + 1):
        for dc in range(-window, window + 1):
            r = center_row + dr
            c = center_column + dc
            if r < 0 or r >= fine.shape[0] or c < 0 or c >= fine.shape[1]:
                continue
            if mask[r, c] == 1:
                continue
            if count < max_points:
                xs[index, count] = c + 0.5
                ys[index, count] = r + 0.5
                values[index, count] = fine[r, c]
                count += 1
    counts[index] = count


def _affected_indices(mask: np.ndarray, coarse_scale: int, coarse_window: int, fine_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return original fine coordinates whose coarse-cell fine window contains cloud."""
    affected = maximum_filter(mask.astype(np.uint8), size=2 * fine_window + 1, mode="nearest") != 0
    rows, columns = np.nonzero(affected)
    coarse_rows = rows // coarse_scale
    coarse_columns = columns // coarse_scale
    keep = (
        (coarse_rows >= coarse_window)
        & (coarse_rows < mask.shape[0] // coarse_scale - coarse_window)
        & (coarse_columns >= coarse_window)
        & (coarse_columns < mask.shape[1] // coarse_scale - coarse_window)
    )
    return rows[keep].astype(np.int32), columns[keep].astype(np.int32)


def _solve_batch(
    rows: np.ndarray, columns: np.ndarray, coarse: np.ndarray, fine: np.ndarray, mask: np.ndarray,
    coarse_scale: int, coarse_window: int, coarse_params: np.ndarray, fine_params: np.ndarray,
    cross_params: np.ndarray, window_sizes: np.ndarray, max_points: int,
) -> np.ndarray:
    """Gather one batch and solve padded cooperative systems with cuSOLVER."""
    batch = len(rows)
    nc = (2 * coarse_window + 1) ** 2
    n = nc + max_points + 2
    d_rows = cp.asarray(rows)
    d_cols = cp.asarray(columns)
    d_fine = cp.asarray(fine, dtype=cp.float64)
    d_mask = cp.asarray(mask, dtype=cp.int8)
    d_window_sizes = cp.asarray(window_sizes, dtype=cp.int32)
    xs = cp.zeros((batch, max_points), dtype=cp.float64)
    ys = cp.zeros_like(xs)
    values = cp.zeros_like(xs)
    counts = cp.zeros(batch, dtype=cp.int32)
    blocks = (batch + 127) // 128
    _gather_candidates[blocks, 128](
        d_rows, d_cols, d_fine, d_mask, d_window_sizes, coarse_scale,
        max_points, xs, ys, values, counts,
    )

    offsets = cp.arange(-coarse_window, coarse_window + 1, dtype=cp.int32)
    off_y, off_x = cp.meshgrid(offsets, offsets, indexing="ij")
    cr = d_rows // coarse_scale
    cc = d_cols // coarse_scale
    coarse_y = (cr[:, None] + off_y.ravel()[None, :]) * coarse_scale + 0.5
    coarse_x = (cc[:, None] + off_x.ravel()[None, :]) * coarse_scale + 0.5
    d_coarse = cp.asarray(coarse, dtype=cp.float64)
    coarse_values = d_coarse[(cr[:, None] + off_y.ravel()[None, :]), (cc[:, None] + off_x.ravel()[None, :])]
    valid = cp.arange(max_points)[None, :] < counts[:, None]

    def gamma(dx, dy, params):
        distance = cp.sqrt(dx * dx + dy * dy)
        return params[0] * (1.0 - cp.exp(-distance / params[1]))

    coarse_p = cp.asarray(coarse_params, dtype=cp.float64)
    fine_p = cp.asarray(fine_params, dtype=cp.float64)
    cross_p = cp.asarray(cross_params, dtype=cp.float64)
    matrix = cp.zeros((batch, n, n), dtype=cp.float64)
    rhs = cp.zeros((batch, n), dtype=cp.float64)
    cc_block = gamma(coarse_x[:, :, None] - coarse_x[:, None, :], coarse_y[:, :, None] - coarse_y[:, None, :], coarse_p)
    ff_block = gamma(xs[:, :, None] - xs[:, None, :], ys[:, :, None] - ys[:, None, :], fine_p)
    cf_distance = cp.sqrt((coarse_x[:, :, None] - xs[:, None, :]) ** 2 + (coarse_y[:, :, None] - ys[:, None, :]) ** 2)
    cf_block = cross_p[0] + cross_p[1] * (1.0 - cp.exp(-cf_distance / cross_p[2]))
    matrix[:, :nc, :nc] = cc_block
    matrix[:, nc:nc + max_points, nc:nc + max_points] = cp.where(valid[:, :, None] & valid[:, None, :], ff_block, 0.0)
    matrix[:, :nc, nc:nc + max_points] = cp.where(valid[:, None, :], cf_block, 0.0)
    matrix[:, nc:nc + max_points, :nc] = cp.swapaxes(matrix[:, :nc, nc:nc + max_points], 1, 2)
    diagonal = cp.arange(nc)
    matrix[:, diagonal, diagonal] = 0.0
    fine_diagonal = nc + cp.arange(max_points)
    matrix[:, fine_diagonal, fine_diagonal] = cp.where(valid, 0.0, 1.0)
    lag_coarse, lag_fine = n - 2, n - 1
    matrix[:, :nc, lag_coarse] = matrix[:, lag_coarse, :nc] = 1.0
    matrix[:, nc:nc + max_points, lag_fine] = valid
    matrix[:, lag_fine, nc:nc + max_points] = valid
    px, py = d_cols + 0.5, d_rows + 0.5
    rhs[:, :nc] = gamma(coarse_x - px[:, None], coarse_y - py[:, None], coarse_p)
    rhs[:, nc:nc + max_points] = cp.where(valid, gamma(xs - px[:, None], ys - py[:, None], fine_p), 0.0)
    rhs[:, lag_coarse] = 1.0
    solution = cp.linalg.solve(matrix, rhs[..., None])[..., 0]
    prediction = cp.sum(solution[:, :nc] * coarse_values, axis=1) + cp.sum(solution[:, nc:nc + max_points] * values, axis=1)
    fallback = d_coarse[cr, cc]
    prediction = cp.where(cp.isfinite(prediction), prediction, fallback)
    return cp.asnumpy(prediction)


def cdsck_coordinate_batched_gpu(
    coarse_scale: int, coarse_window: int, fine_window: int, coarse: np.ndarray, fine: np.ndarray,
    mask: np.ndarray, weights: np.ndarray, coarse_params: np.ndarray, fine_params: np.ndarray,
    cross_params: np.ndarray, max_points: int, max_radius: int, batch_size: int,
) -> np.ndarray:
    """Fast fixed DSCK plus batched cloud-aware cooperative corrections."""
    prediction = dsck_coordinate_gpu(coarse_scale, coarse_window, fine_window, coarse, fine, weights)
    rows, columns = _affected_indices(mask, coarse_scale, coarse_window, fine_window)
    window_sizes = _adaptive_window_sizes(
        mask, coarse_scale, coarse_window, fine_window,
        target_points=50, max_window=min(max_radius, 41),
    )
    for start in range(0, len(rows), batch_size):
        part = slice(start, start + batch_size)
        prediction[rows[part], columns[part]] = _solve_batch(
            rows[part], columns[part], coarse, fine, mask, coarse_scale, coarse_window,
            coarse_params, fine_params, cross_params, window_sizes, max_points,
        )
    return prediction
