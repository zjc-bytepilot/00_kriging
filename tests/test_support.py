"""Equivalence checks for extracted support-scale numerical kernels."""

from __future__ import annotations

import unittest

import numpy as np

from kriging import dsck, spatial, support


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

    def test_support_owns_atprk_deconvolution_kernel(self) -> None:
        self.assertEqual(support.atprk_deconvolution.__module__, "kriging.support")

    def test_support_owns_dsck_deconvolution_kernels(self) -> None:
        self.assertEqual(support.deconvolution_fine.__module__, "kriging.support")
        actual = support.deconvolution_fine(
            1,
            2,
            np.array([2.0, 4.0]),
            0.5,
            0.5,
            1,
            1,
            0.1,
        )

        np.testing.assert_allclose(actual, np.array([1.2, 2.4]), rtol=0, atol=0)

    def test_dsck_imports_canonical_spatial_primitives(self) -> None:
        self.assertIs(dsck.extend_plane, spatial.extend_plane)
        self.assertIs(dsck.PSF, spatial.PSF)
        self.assertIs(dsck.downsample_plane, spatial.downsample_plane)
        self.assertIs(dsck.myfun, spatial.myfun)
        self.assertIs(dsck.myfun2, spatial.myfun2)
        self.assertIs(dsck.semivariogram, spatial.semivariogram)
        self.assertIs(dsck.semivariogram_cross, spatial.semivariogram_cross)

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
