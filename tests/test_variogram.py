"""Tests for typed self- and cross-variogram fitting services."""

from __future__ import annotations

import unittest

import numpy as np

from kriging import spatial
from kriging.variogram import (
    CrossVariogramEstimator,
    ExponentialCrossVariogramModel,
    ExponentialVariogramModel,
    VariogramEstimator,
)


class VariogramServiceTest(unittest.TestCase):
    """Ensure the new service preserves the established numerical primitives."""

    def test_self_model_matches_legacy_exponential_function(self) -> None:
        distances = np.array([0.0, 1.0, 3.0])
        parameters = np.array([2.0, 4.0])

        actual = ExponentialVariogramModel().evaluate(parameters, distances)
        expected = spatial.exponential_variogram(parameters, distances)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_cross_model_matches_legacy_exponential_function(self) -> None:
        distances = np.array([0.0, 1.0, 3.0])
        parameters = np.array([0.5, 2.0, 4.0])

        actual = ExponentialCrossVariogramModel().evaluate(parameters, distances)
        expected = spatial.exponential_cross_variogram(parameters, distances)

        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_self_estimator_records_lags_and_empirical_values(self) -> None:
        plane = np.arange(16.0).reshape(4, 4)
        distances = np.array([3.0, 6.0])

        fit = VariogramEstimator().fit(
            plane,
            max_lag=2,
            distances=distances,
            initial=np.array([100.0, 1.0]),
        )

        np.testing.assert_array_equal(fit.lags, distances)
        np.testing.assert_allclose(
            fit.empirical_values,
            np.array([spatial.semivariogram(plane, 1), spatial.semivariogram(plane, 2)]),
            rtol=0,
            atol=0,
        )
        self.assertEqual(fit.parameters.shape, (2,))

    def test_self_estimator_uses_injected_empirical_kernel(self) -> None:
        calls: list[int] = []

        def empirical_kernel(plane: np.ndarray, lag: int) -> float:
            calls.append(lag)
            return float(lag * 10)

        fit = VariogramEstimator(empirical_kernel=empirical_kernel).fit(
            np.zeros((3, 3)),
            max_lag=2,
            distances=np.array([1.0, 2.0]),
            initial=np.array([1.0, 1.0]),
        )

        self.assertEqual(calls, [1, 2])
        np.testing.assert_array_equal(fit.empirical_values, np.array([10.0, 20.0]))

    def test_self_estimator_uses_injected_residual_kernel(self) -> None:
        calls: list[np.ndarray] = []

        def residual_kernel(parameters: np.ndarray, distances: np.ndarray, observed: np.ndarray) -> np.ndarray:
            calls.append(parameters.copy())
            return parameters[:1] - observed

        VariogramEstimator(residual_kernel=residual_kernel).fit(
            np.zeros((3, 3)),
            max_lag=1,
            distances=np.array([1.0]),
            initial=np.array([1.0, 1.0]),
        )

        self.assertGreater(len(calls), 0)

    def test_estimator_uses_model_residual_without_override(self) -> None:
        class RecordingSelfModel:
            def __init__(self) -> None:
                self.calls = 0

            def residual(
                self,
                parameters: np.ndarray,
                distances: np.ndarray,
                observed: np.ndarray,
            ) -> np.ndarray:
                self.calls += 1
                return np.full(observed.shape, parameters[0]) - observed

        model = RecordingSelfModel()
        VariogramEstimator(model=model).fit(
            np.zeros((3, 3)),
            max_lag=1,
            distances=np.array([1.0]),
            initial=np.array([1.0, 1.0]),
        )

        self.assertGreater(model.calls, 0)

    def test_cross_estimator_records_lags_and_empirical_values(self) -> None:
        first = np.arange(16.0).reshape(4, 4)
        second = first * 2.0
        distances = np.array([3.0, 6.0])

        fit = CrossVariogramEstimator().fit_cross(
            first,
            second,
            max_lag=2,
            distances=distances,
            initial=np.array([10.0, 100.0, 1.0]),
        )

        np.testing.assert_array_equal(fit.lags, distances)
        np.testing.assert_allclose(
            fit.empirical_values,
            np.array(
                [
                    spatial.cross_semivariogram(first, second, 1),
                    spatial.cross_semivariogram(first, second, 2),
                ]
            ),
            rtol=0,
            atol=0,
        )
        self.assertEqual(fit.parameters.shape, (3,))

    def test_cross_estimator_uses_its_self_model_for_self_fit(self) -> None:
        class RecordingSelfModel:
            def __init__(self) -> None:
                self.calls = 0

            def residual(
                self,
                parameters: np.ndarray,
                distances: np.ndarray,
                observed: np.ndarray,
            ) -> np.ndarray:
                self.calls += 1
                return np.full(observed.shape, parameters[0]) - observed

        self_model = RecordingSelfModel()
        fit = CrossVariogramEstimator(self_model=self_model).fit(
            np.arange(16.0).reshape(4, 4),
            max_lag=2,
            distances=np.array([1.0, 2.0]),
            initial=np.array([1.0, 1.0]),
        )

        self.assertEqual(fit.parameters.shape, (2,))
        self.assertGreater(self_model.calls, 0)

    def test_cross_estimator_retains_legacy_positional_constructor(self) -> None:
        cross_model = ExponentialCrossVariogramModel()
        estimator = CrossVariogramEstimator(
            cross_model,
            spatial.semivariogram,
            spatial.cross_semivariogram,
            spatial.exponential_cross_variogram_residual,
        )
        fit = estimator.fit_cross(
            np.arange(16.0).reshape(4, 4),
            np.arange(16.0).reshape(4, 4),
            max_lag=2,
            distances=np.array([1.0, 2.0]),
            initial=np.array([1.0, 1.0, 1.0]),
        )

        self.assertIs(estimator.model, cross_model)
        self.assertEqual(fit.parameters.shape, (3,))

    def test_cross_estimator_uses_injected_empirical_kernel(self) -> None:
        calls: list[int] = []

        def empirical_kernel(first: np.ndarray, second: np.ndarray, lag: int) -> float:
            calls.append(lag)
            return float(lag * 10)

        fit = CrossVariogramEstimator(cross_empirical_kernel=empirical_kernel).fit_cross(
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            max_lag=2,
            distances=np.array([1.0, 2.0]),
            initial=np.array([1.0, 1.0, 1.0]),
        )

        self.assertEqual(calls, [1, 2])
        np.testing.assert_array_equal(fit.empirical_values, np.array([10.0, 20.0]))
