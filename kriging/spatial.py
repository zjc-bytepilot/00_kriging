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


@jit(nopython=True)
def exponential_cross_variogram(parameters, distance):
    """Exponential cross-semivariogram with a constant term."""
    return parameters[0] + parameters[1] * (1 - np.exp(-distance / parameters[2]))


def exponential_cross_variogram_residual(parameters, distance, observed):
    return exponential_cross_variogram(parameters, distance) - observed


# Compatibility aliases used by the numerical kernels.
myfun = exponential_variogram
myfun_fit = exponential_variogram_residual
myfun2 = exponential_cross_variogram
myfun2_fit = exponential_cross_variogram_residual


@jit(nopython=True)
def semivariogram(plane, lag):
    """Estimate the omnidirectional semivariogram for one lag."""
    rows, columns = plane.shape
    total = 0.0
    count = 0
    for row in range(lag, rows):
        for column in range(columns):
            total += (plane[row, column] - plane[row - lag, column]) ** 2
            count += 1
    for row in range(rows - lag):
        for column in range(columns):
            total += (plane[row, column] - plane[row + lag, column]) ** 2
            count += 1
    for row in range(rows):
        for column in range(lag, columns):
            total += (plane[row, column] - plane[row, column - lag]) ** 2
            count += 1
    for row in range(rows):
        for column in range(columns - lag):
            total += (plane[row, column] - plane[row, column + lag]) ** 2
            count += 1
    return total / (2 * count)


@jit(nopython=True)
def cross_semivariogram(first, second, lag):
    """Estimate the omnidirectional cross-semivariogram for one lag."""
    rows, columns = first.shape
    total = 0.0
    count = 0
    for row in range(lag, rows):
        for column in range(columns):
            total += ((first[row, column] - first[row - lag, column])
                      * (second[row, column] - second[row - lag, column]))
            count += 1
    for row in range(rows - lag):
        for column in range(columns):
            total += ((first[row, column] - first[row + lag, column])
                      * (second[row, column] - second[row + lag, column]))
            count += 1
    for row in range(rows):
        for column in range(lag, columns):
            total += ((first[row, column] - first[row, column - lag])
                      * (second[row, column] - second[row, column - lag]))
            count += 1
    for row in range(rows):
        for column in range(columns - lag):
            total += ((first[row, column] - first[row, column + lag])
                      * (second[row, column] - second[row, column + lag]))
            count += 1
    return total / (2 * count)


semivariogram_cross = cross_semivariogram
