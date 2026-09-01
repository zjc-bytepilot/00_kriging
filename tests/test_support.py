"""Equivalence checks for extracted support-scale numerical kernels."""

from __future__ import annotations

import unittest

import numpy as np

from kriging import spatial, support


class SupportKernelTest(unittest.TestCase):
    """Preserve the exact trial-and-error support transformations."""

    def test_support_reexports_stable_psf_and_downsampling_primitives(self) -> None:
        self.assertIs(support.gaussian_psf, spatial.gaussian_psf)
        self.assertIs(support.downsample_plane, spatial.downsample_plane)

    def test_atprk_deconvolution_matches_pre_extraction_reference(self) -> None:
        actual = support.atprk_deconvolution(
            1,
            3,
            np.array([2.0, 4.0]),
            0.5,
            0.5,
            1,
            1,
            0.1,
        )

        np.testing.assert_allclose(actual, np.array([1.2, 2.4]), rtol=0, atol=0)

    def test_dsck_cross_deconvolution_matches_pre_extraction_reference(self) -> None:
        actual = support.dsck_deconvolution_cross(
            1,
            3,
            2,
            np.array([1.0, 2.0, 4.0]),
            0.5,
            0.5,
            0.5,
            1,
            1,
            1,
            0.1,
        )

        np.testing.assert_allclose(actual, np.array([0.6, 1.2, 2.4]), rtol=0, atol=0)
