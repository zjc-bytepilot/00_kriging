"""Typed services for fitting the project's existing variogram models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from . import spatial

EmpiricalKernel = Callable[[np.ndarray, int], float]
CrossEmpiricalKernel = Callable[[np.ndarray, np.ndarray, int], float]
ResidualKernel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class VariogramFit:
    """Parameters and observations produced by one variogram fit."""

    parameters: np.ndarray
    lags: np.ndarray
    empirical_values: np.ndarray


class ExponentialVariogramModel:
    """Adapter for the legacy two-parameter exponential variogram."""

    @staticmethod
    def evaluate(parameters: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return spatial.exponential_variogram(parameters, distances)

    @staticmethod
    def residual(parameters: np.ndarray, distances: np.ndarray, observed: np.ndarray) -> np.ndarray:
        return spatial.exponential_variogram_residual(parameters, distances, observed)


class ExponentialCrossVariogramModel:
    """Adapter for the legacy three-parameter exponential cross-variogram."""

    @staticmethod
    def evaluate(parameters: np.ndarray, distances: np.ndarray) -> np.ndarray:
        return spatial.exponential_cross_variogram(parameters, distances)

    @staticmethod
    def residual(parameters: np.ndarray, distances: np.ndarray, observed: np.ndarray) -> np.ndarray:
        return spatial.exponential_cross_variogram_residual(parameters, distances, observed)


class VariogramEstimator:
    """Estimate and fit the project's self-semivariogram without changing its kernel."""

    def __init__(
        self,
        model: ExponentialVariogramModel | None = None,
        empirical_kernel: EmpiricalKernel = spatial.semivariogram,
        residual_kernel: ResidualKernel = spatial.exponential_variogram_residual,
    ) -> None:
        self.model = model or ExponentialVariogramModel()
        self._empirical_kernel = empirical_kernel
        self._residual_kernel = residual_kernel

    def empirical(self, plane: np.ndarray, max_lag: int) -> np.ndarray:
        return np.asarray([self._empirical_kernel(plane, lag) for lag in range(1, max_lag + 1)])

    def fit(
        self,
        plane: np.ndarray,
        max_lag: int,
        distances: np.ndarray,
        initial: np.ndarray,
    ) -> VariogramFit:
        empirical_values = self.empirical(plane, max_lag)
        result = least_squares(self._residual_kernel, initial, args=(distances, empirical_values))
        return VariogramFit(
            parameters=np.asarray(result.x),
            lags=np.asarray(distances),
            empirical_values=empirical_values,
        )


class CrossVariogramEstimator(VariogramEstimator):
    """Add cross-semivariogram fitting while retaining the self-fit interface."""

    def __init__(
        self,
        model: ExponentialCrossVariogramModel | None = None,
        empirical_kernel: EmpiricalKernel = spatial.semivariogram,
        cross_empirical_kernel: CrossEmpiricalKernel = spatial.cross_semivariogram,
        residual_kernel: ResidualKernel = spatial.exponential_cross_variogram_residual,
    ) -> None:
        self.model = model or ExponentialCrossVariogramModel()
        self._empirical_kernel = empirical_kernel
        self._cross_empirical_kernel = cross_empirical_kernel
        self._residual_kernel = residual_kernel

    def empirical_cross(self, first: np.ndarray, second: np.ndarray, max_lag: int) -> np.ndarray:
        return np.asarray(
            [self._cross_empirical_kernel(first, second, lag) for lag in range(1, max_lag + 1)]
        )

    def fit_cross(
        self,
        first: np.ndarray,
        second: np.ndarray,
        max_lag: int,
        distances: np.ndarray,
        initial: np.ndarray,
    ) -> VariogramFit:
        empirical_values = self.empirical_cross(first, second, max_lag)
        result = least_squares(self._residual_kernel, initial, args=(distances, empirical_values))
        return VariogramFit(
            parameters=np.asarray(result.x),
            lags=np.asarray(distances),
            empirical_values=empirical_values,
        )
