"""Support-scale regularization and deconvolution numerical kernels."""

from __future__ import annotations

import numpy as np
from numba import jit

from .spatial import (
    downsample_plane,
    exponential_cross_variogram,
    exponential_variogram,
    extend_plane,
    gaussian_psf,
)


@jit(nopython=True)
def atprk_regularization(h, scale, parameters):
    assume_line = np.zeros((h + 1, 1))
    line_rows, line_columns = np.where(assume_line == 0)
    assume_area = np.zeros((scale, scale))
    area_rows, area_columns = np.where(assume_area == 0)
    regularized = np.zeros((h + 1, 1))
    for lag in range(h + 1):
        regularized[lag, 0] = 0
        for first in range(scale ** 2):
            for second in range(scale ** 2):
                point_first = np.array([
                    line_rows[lag] * scale + area_rows[first] + 0.5,
                    line_columns[lag] * scale + area_columns[first] + 0.5,
                ])
                point_second = np.array([
                    line_rows[0] * scale + area_rows[second] + 0.5,
                    line_columns[0] * scale + area_columns[second] + 0.5,
                ])
                regularized[lag, 0] += exponential_variogram(
                    parameters, np.sqrt(np.sum((point_first - point_second) ** 2))
                )
    return regularized / scale ** 4


@jit(nopython=True)
def atprk_deconvolution(h, scale, area_parameters, sill_min, range_min, sill_steps, range_steps, rate):
    area_variogram = exponential_variogram(area_parameters, np.arange(1, scale * h + 1))
    area_samples = area_variogram[scale - 1::scale]
    difference_min = 10 ** 6
    for sill_step in range(1, sill_steps + 1):
        for range_step in range(1, range_steps + 1):
            candidate = np.array([
                (sill_min + rate * sill_step) * area_parameters[0],
                (range_min + rate * range_step) * area_parameters[1],
            ])
            regularized = atprk_regularization(h, scale, candidate)
            difference = np.linalg.norm(
                regularized[1:h + 1, 0] - regularized[0, 0] - area_samples
            )
            if difference <= difference_min:
                best = candidate
                difference_min = difference
    return best


@jit(nopython=True)
def regularization_coarse(h, coarse_scale, fine_scale, parameters):
    assume_line = np.zeros((h + 1, 1))
    line_rows, line_columns = np.where(assume_line == 0)
    assume_area = np.zeros((coarse_scale, coarse_scale))
    area_rows, area_columns = np.where(assume_area == 0)
    regularized = np.zeros((h + 1, 1))
    for lag in range(h + 1):
        regularized[lag] = 0
        for first in range(coarse_scale ** 2):
            for second in range(coarse_scale ** 2):
                point_first = np.array([
                    line_rows[lag] * fine_scale * coarse_scale + fine_scale * area_rows[first] + 1,
                    line_columns[lag] * fine_scale * coarse_scale + fine_scale * area_columns[first] + 1,
                ])
                point_second = np.array([
                    line_rows[0] * fine_scale * coarse_scale + fine_scale * area_rows[second] + 1,
                    line_columns[0] * fine_scale * coarse_scale + fine_scale * area_columns[second] + 1,
                ])
                regularized[lag] += exponential_variogram(
                    parameters, np.sqrt(np.sum((point_first - point_second) ** 2))
                )
    return regularized / coarse_scale ** 4


@jit(nopython=True)
def deconvolution_coarse(h, coarse_scale, fine_scale, area_parameters, sill_min, range_min, sill_steps, range_steps, rate):
    area_variogram = exponential_variogram(
        area_parameters, np.arange(1, fine_scale * coarse_scale * h + 1)
    )
    area_samples = area_variogram[fine_scale * coarse_scale - 1::fine_scale * coarse_scale].T
    difference_min = 1e6
    for sill_step in range(1, sill_steps + 1):
        for range_step in range(1, range_steps + 1):
            candidate = np.array([
                (sill_min + rate * sill_step) * area_parameters[0],
                (range_min + rate * range_step) * area_parameters[1],
            ])
            regularized = regularization_coarse(h, coarse_scale, fine_scale, candidate)
            difference = np.linalg.norm(regularized[1:h + 1] - regularized[0] - area_samples)
            if difference <= difference_min:
                best = candidate
                difference_min = difference
    return best


@jit(nopython=True)
def regularization_cross(h, coarse_scale, fine_scale, parameters):
    assume_line = np.zeros((h + 1, 1))
    line_rows, line_columns = np.where(assume_line == 0)
    coarse_area = np.zeros((coarse_scale, coarse_scale))
    coarse_rows, coarse_columns = np.where(coarse_area == 0)
    fine_area = np.zeros((fine_scale * coarse_scale, fine_scale * coarse_scale))
    fine_rows, fine_columns = np.where(fine_area == 0)
    regularized = np.zeros((h + 1, 1))
    for lag in range(h + 1):
        regularized[lag] = 0
        for first in range(coarse_scale ** 2):
            for second in range((fine_scale * coarse_scale) ** 2):
                point_first = np.array([
                    line_rows[lag] * fine_scale * coarse_scale + fine_rows[second] + 0.5,
                    line_columns[lag] * fine_scale * coarse_scale + fine_columns[second] + 0.5,
                ])
                point_second = np.array([
                    line_rows[0] * fine_scale * coarse_scale + fine_scale * coarse_rows[first] + 1,
                    line_columns[0] * fine_scale * coarse_scale + fine_scale * coarse_columns[first] + 1,
                ])
                regularized[lag] += exponential_cross_variogram(
                    parameters, np.sqrt(np.sum((point_first - point_second) ** 2))
                )
    return regularized / (coarse_scale * fine_scale * coarse_scale) ** 2


@jit(nopython=True)
def deconvolution_cross(h, coarse_scale, fine_scale, area_parameters, constant_min, sill_min, range_min, sill_steps, range_steps, constant_steps, rate):
    area_variogram = exponential_cross_variogram(
        area_parameters,
        np.arange(fine_scale * coarse_scale, fine_scale * coarse_scale * h + 1, fine_scale * coarse_scale),
    )
    difference_min = 1e6
    for constant_step in range(1, constant_steps + 1):
        for sill_step in range(1, sill_steps + 1):
            for range_step in range(1, range_steps + 1):
                candidate = np.array([
                    (constant_min + rate * constant_step) * area_parameters[0],
                    (sill_min + rate * sill_step) * area_parameters[1],
                    (range_min + rate * range_step) * area_parameters[2],
                ])
                regularized = regularization_cross(h, coarse_scale, fine_scale, candidate)
                difference = np.sqrt(np.sum(
                    (regularized[1:h + 1, 0] - regularized[0, 0] - area_variogram.T) ** 2
                ))
                if difference <= difference_min:
                    best = candidate
                    difference_min = difference
    return best


@jit(nopython=True)
def regularization_fine(h, fine_scale, parameters):
    assume_line = np.zeros((h + 1, 1))
    line_rows, line_columns = np.where(assume_line == 0)
    assume_area = np.zeros((fine_scale, fine_scale))
    area_rows, area_columns = np.where(assume_area == 0)
    regularized = np.zeros((h + 1, 1))
    for lag in range(h + 1):
        regularized[lag] = 0
        for first in range(fine_scale ** 2):
            for second in range(fine_scale ** 2):
                point_first = np.array([
                    line_rows[lag] * fine_scale + area_rows[first] + 0.5,
                    line_columns[lag] * fine_scale + area_columns[first] + 0.5,
                ])
                point_second = np.array([
                    line_rows[0] * fine_scale + area_rows[second] + 0.5,
                    line_columns[0] * fine_scale + area_columns[second] + 0.5,
                ])
                regularized[lag] += exponential_variogram(
                    parameters, np.sqrt(np.sum((point_first - point_second) ** 2))
                )
    return regularized / fine_scale ** 4


@jit(nopython=True)
def deconvolution_fine(h, fine_scale, area_parameters, sill_min, range_min, sill_steps, range_steps, rate):
    area_variogram = exponential_variogram(area_parameters, np.arange(1, fine_scale * h + 1))
    area_samples = area_variogram[fine_scale - 1::fine_scale]
    difference_min = 10 ** 6
    for sill_step in range(1, sill_steps + 1):
        for range_step in range(1, range_steps + 1):
            candidate = np.array([
                (sill_min + rate * sill_step) * area_parameters[0],
                (range_min + rate * range_step) * area_parameters[1],
            ])
            regularized = regularization_fine(h, fine_scale, candidate)
            difference = np.linalg.norm(
                regularized[1:h + 1] - regularized[0] - area_samples
            )
            if difference <= difference_min:
                best = candidate
                difference_min = difference
    return best


# Compatibility names retained for existing direct callers.
dsck_deconvolution_coarse = deconvolution_coarse
dsck_deconvolution_cross = deconvolution_cross
dsck_deconvolution_fine = deconvolution_fine

__all__ = [
    "atprk_deconvolution",
    "atprk_regularization",
    "deconvolution_coarse",
    "deconvolution_cross",
    "deconvolution_fine",
    "downsample_plane",
    "dsck_deconvolution_coarse",
    "dsck_deconvolution_cross",
    "dsck_deconvolution_fine",
    "extend_plane",
    "gaussian_psf",
    "regularization_coarse",
    "regularization_cross",
    "regularization_fine",
]
