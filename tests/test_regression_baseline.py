"""Numerical regression checks captured before architecture refactoring."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from kriging.config import ATPRKConfig, DSCKConfig, SearchConfig
from kriging.estimators import ATPRKInterpolator, DSCKInterpolator
from tests.test_synthetic_smoke import synthetic_pair


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "kriging_regression_baseline.npz"


def load_regression_baseline() -> dict[str, np.ndarray]:
    """Load the immutable output reference captured from the public API."""
    with np.load(FIXTURE_PATH) as values:
        return {name: values[name] for name in values.files}


class NumericalRegressionBaselineTest(unittest.TestCase):
    """Protect public results while internal responsibilities are moved."""

    def _inputs(self) -> tuple[np.ndarray, np.ndarray, SearchConfig]:
        coarse, fine = synthetic_pair()
        search = SearchConfig(
            constant_min=0.5,
            sill_min=0.5,
            range_min=0.5,
            sill_steps=1,
            range_steps=1,
            constant_steps=1,
            step_size=0.1,
            max_lag=1,
        )
        return coarse, fine, search

    def test_atprk_output_matches_locked_numerical_baseline(self) -> None:
        baseline = load_regression_baseline()
        coarse, fine, search = self._inputs()
        atprk = ATPRKInterpolator(ATPRKConfig(window=1, psf_sigma=1.0), search)
        atprk_result = atprk.sharpen(coarse, fine, band_count=1)

        np.testing.assert_allclose(
            atprk_result.prediction,
            baseline["atprk_prediction"],
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            atprk_result.uncertainty,
            baseline["atprk_uncertainty"],
            rtol=1e-8,
            atol=1e-10,
        )

    def test_dsck_output_matches_locked_numerical_baseline(self) -> None:
        baseline = load_regression_baseline()
        coarse, fine, search = self._inputs()
        dsck = DSCKInterpolator(
            DSCKConfig(
                coarse_scale=3,
                fine_scale=2,
                coarse_window=1,
                fine_window=1,
                psf_sigma=1.0,
            ),
            search,
        )
        dsck_prediction = dsck.sharpen(coarse, fine, band_count=1)
        np.testing.assert_allclose(
            dsck_prediction,
            baseline["dsck_prediction"],
            rtol=1e-8,
            atol=1e-10,
        )
