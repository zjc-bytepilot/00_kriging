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
