"""Shared spatial degradation and variogram primitives.

These functions intentionally remain stateless: Numba can compile pure numerical
kernels more reliably than bound class methods.
"""

from __future__ import annotations

import numpy as np
from numba import jit


def extend_plane(plane: np.ndarray, width: int) -> np.ndarray:
    """Pad a plane by repeating its edge values."""
    if plane.ndim != 2:
        raise ValueError("plane 必须是二维数组。")
    if width < 0:
        raise ValueError("width 不能小于 0。")
    return np.pad(plane, width, mode="edge") if width else plane.copy()


def gaussian_psf(scale: int, window: int, sigma: float) -> np.ndarray:
    """Build the normalized Gaussian point-spread function used by both models."""
    if min(scale, window) <= 0 or sigma <= 0:
        raise ValueError("scale、window 和 sigma 必须大于 0。")
    size = (2 * window + 1) * scale
    coordinates = np.arange(size, dtype=float) + 0.5
    center = size / 2
    row_distance, column_distance = np.meshgrid(
        coordinates - center, coordinates - center, indexing="ij"
    )
    kernel = np.exp(-(row_distance ** 2 + column_distance ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


# Backward-compatible public name used by the original implementation.
PSF = gaussian_psf


def downsample_plane(
    plane: np.ndarray,
    scale: int,
    window: int,
    psf: np.ndarray,
) -> np.ndarray:
    """Degrade a fine plane using the supplied PSF and integer scale."""
    padded = extend_plane(plane, window * scale)
    rows, columns = padded.shape
    output = np.zeros((rows // scale, columns // scale))
    for row in range(window * scale, rows - window * scale, scale):
        for column in range(window * scale, columns - window * scale, scale):
            output_row = (row + scale - 1) // scale
            output_column = (column + scale - 1) // scale
            local_window = padded[
                row - window * scale:row + window * scale + scale,
                column - window * scale:column + window * scale + scale,
            ]
            output[output_row, output_column] = np.sum(local_window * psf)
    return output[window:-window, window:-window]


@jit(nopython=True)
def exponential_variogram(parameters, distance):
    """Exponential semivariogram without a constant term."""
    return parameters[0] * (1 - np.exp(-distance / parameters[1]))


def exponential_variogram_residual(parameters, distance, observed):
    return exponential_variogram(parameters, distance) - observed


def legacy_exponential_variogram_residual(parameters, distance, observed):
    """Preserve DSCK's historical NumPy residual operation order."""
    return parameters[0] * (1 - np.exp(-distance / parameters[1])) - observed


@jit(nopython=True)
def exponential_cross_variogram(parameters, distance):
    """Exponential cross-semivariogram with a constant term."""
    return parameters[0] + parameters[1] * (1 - np.exp(-distance / parameters[2]))


def exponential_cross_variogram_residual(parameters, distance, observed):
    return exponential_cross_variogram(parameters, distance) - observed


def legacy_exponential_cross_variogram_residual(parameters, distance, observed):
    """Preserve DSCK's historical NumPy cross-residual operation order."""
    return parameters[0] + parameters[1] * (1 - np.exp(-distance / parameters[2])) - observed


# Compatibility aliases used by the numerical kernels.
myfun = exponential_variogram
myfun_fit = legacy_exponential_variogram_residual
myfun2 = exponential_cross_variogram
myfun2_fit = legacy_exponential_cross_variogram_residual


@jit(nopython=True)
def semivariogram(plane, lag):
    """Estimate the omnidirectional semivariogram for one lag."""
    rows, columns = plane.shape
    vertical_backward_total = 0.0
    vertical_backward_count = 0
    for row in range(lag, rows):
        for column in range(columns):
            vertical_backward_total += (plane[row, column] - plane[row - lag, column]) ** 2
            vertical_backward_count += 1
    vertical_forward_total = 0.0
    vertical_forward_count = 0
    for row in range(rows - lag):
        for column in range(columns):
            vertical_forward_total += (plane[row, column] - plane[row + lag, column]) ** 2
            vertical_forward_count += 1
    horizontal_backward_total = 0.0
    horizontal_backward_count = 0
    for row in range(rows):
        for column in range(lag, columns):
            horizontal_backward_total += (plane[row, column] - plane[row, column - lag]) ** 2
            horizontal_backward_count += 1
    horizontal_forward_total = 0.0
    horizontal_forward_count = 0
    for row in range(rows):
        for column in range(columns - lag):
            horizontal_forward_total += (plane[row, column] - plane[row, column + lag]) ** 2
            horizontal_forward_count += 1
    total = (
        vertical_backward_total
        + vertical_forward_total
        + horizontal_backward_total
        + horizontal_forward_total
    )
    count = (
        vertical_backward_count
        + vertical_forward_count
        + horizontal_backward_count
        + horizontal_forward_count
    )
    return total / (2 * count)


@jit(nopython=True)
def cross_semivariogram(first, second, lag):
    """Estimate the omnidirectional cross-semivariogram for one lag."""
    rows, columns = first.shape
    vertical_backward_total = 0.0
    vertical_backward_count = 0
    for row in range(lag, rows):
        for column in range(columns):
            vertical_backward_total += (
                (first[row, column] - first[row - lag, column])
                * (second[row, column] - second[row - lag, column])
            )
            vertical_backward_count += 1
    vertical_forward_total = 0.0
    vertical_forward_count = 0
    for row in range(rows - lag):
        for column in range(columns):
            vertical_forward_total += (
                (first[row, column] - first[row + lag, column])
                * (second[row, column] - second[row + lag, column])
            )
            vertical_forward_count += 1
    horizontal_backward_total = 0.0
    horizontal_backward_count = 0
    for row in range(rows):
        for column in range(lag, columns):
            horizontal_backward_total += (
                (first[row, column] - first[row, column - lag])
                * (second[row, column] - second[row, column - lag])
            )
            horizontal_backward_count += 1
    horizontal_forward_total = 0.0
    horizontal_forward_count = 0
    for row in range(rows):
        for column in range(columns - lag):
            horizontal_forward_total += (
                (first[row, column] - first[row, column + lag])
                * (second[row, column] - second[row, column + lag])
            )
            horizontal_forward_count += 1
    total = (
        vertical_backward_total
        + vertical_forward_total
        + horizontal_backward_total
        + horizontal_forward_total
    )
    count = (
        vertical_backward_count
        + vertical_forward_count
        + horizontal_backward_count
        + horizontal_forward_count
    )
    return total / (2 * count)


semivariogram_cross = cross_semivariogram
