"""Public API integration checks for the refactored algorithm services."""

from __future__ import annotations

import unittest

from kriging.config import ATPRKConfig, SearchConfig
from kriging.estimators import ATPRKInterpolator


class AlgorithmIntegrationTest(unittest.TestCase):
    """Check reusable configuration-dependent objects at the public boundary."""

    def test_atprk_reuses_cached_psf_for_equal_scale(self) -> None:
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
        model = ATPRKInterpolator(ATPRKConfig(window=1, psf_sigma=1.0), search)

        first = model._psf_for_scale(3)
        second = model._psf_for_scale(3)

        self.assertIs(first, second)
