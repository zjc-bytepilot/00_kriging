"""Small end-to-end tests using generated spatial data."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from kriging.config import ATPRKConfig, DSCKConfig, SearchConfig
from kriging.data import GeoTiffLoader, save_geotiff
from kriging.estimators import ATPRKInterpolator, DSCKInterpolator
from kriging.metrics import evaluate_spectral
from kriging.spatial import downsample_plane, gaussian_psf


def synthetic_pair(size: int = 15, scale: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Create correlated fine and coarse single-band cubes."""
    generator = np.random.default_rng(42)
    fine_plane = gaussian_filter(generator.normal(size=(size, size)), sigma=1.2)
    fine_plane = (fine_plane - fine_plane.min()) / (fine_plane.max() - fine_plane.min())
    psf = gaussian_psf(scale=scale, window=1, sigma=1.0)
    coarse_plane = downsample_plane(fine_plane, scale=scale, window=1, psf=psf)
    return coarse_plane[:, :, None], fine_plane[:, :, None]


class SyntheticKrigingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coarse, cls.fine = synthetic_pair()
        cls.search = SearchConfig(
            constant_min=0.5,
            sill_min=0.5,
            range_min=0.5,
            sill_steps=1,
            range_steps=1,
            constant_steps=1,
            step_size=0.1,
            max_lag=1,
        )

    def assert_valid_prediction(self, prediction: np.ndarray) -> None:
        self.assertEqual(prediction.shape, self.fine.shape)
        self.assertTrue(np.all(np.isfinite(prediction)))
        metrics = evaluate_spectral(self.fine, prediction, scale=3)
        self.assertTrue(np.all(np.isfinite(metrics.rmse)))
        self.assertTrue(np.isfinite(metrics.ergas))

    def test_dsck_end_to_end(self) -> None:
        model = DSCKInterpolator(
            DSCKConfig(
                coarse_scale=3,
                fine_scale=2,
                coarse_window=1,
                fine_window=1,
                psf_sigma=1.0,
            ),
            self.search,
        )
        prediction = model.sharpen(self.coarse, self.fine, band_count=1)
        self.assert_valid_prediction(prediction)

    def test_atprk_end_to_end(self) -> None:
        model = ATPRKInterpolator(ATPRKConfig(window=1, psf_sigma=1.0), self.search)
        result = model.sharpen(self.coarse, self.fine, band_count=1)
        self.assertEqual(result.uncertainty.shape, self.fine.shape)
        self.assertTrue(np.all(np.isfinite(result.uncertainty)))
        self.assert_valid_prediction(result.prediction)

    def test_geotiff_round_trip_preserves_spatial_metadata(self) -> None:
        project_tmp = Path(__file__).resolve().parents[1] / "tmp"
        project_tmp.mkdir(exist_ok=True)
        profile = dict(
            driver="GTiff",
            height=self.fine.shape[0],
            width=self.fine.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:32650",
            transform=from_origin(500000, 4000000, 10, 10),
        )
        source_path = project_tmp / "Ssynthetic_test.tif"
        save_geotiff(source_path, self.fine, profile)
        loaded = GeoTiffLoader(project_tmp, "S{identifier}.tif").load(["synthetic_test"])[0]
        self.assertTrue(np.allclose(loaded.values, self.fine))
        self.assertEqual(loaded.profile["crs"], rasterio.crs.CRS.from_epsg(32650))
        self.assertEqual(loaded.profile["transform"], profile["transform"])


if __name__ == "__main__":
    unittest.main()
