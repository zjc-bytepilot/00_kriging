"""Configuration-driven experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import ExperimentConfig
from .data import GeoTiffLoader, save_geotiff
from .estimators import ATPRKInterpolator, DSCKInterpolator
from .metrics import SpectralMetrics, evaluate_spectral


@dataclass(frozen=True)
class MethodResult:
    prediction: np.ndarray
    metrics: SpectralMetrics | None
    elapsed_seconds: float
    uncertainty: np.ndarray | None = None


class KrigingExperiment:
    """Load configured data and run one or more downscaling methods."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._reference_profile: dict | None = None

    def load_first_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        data = self.config.data
        identifiers = data.dates
        coarse_raster = GeoTiffLoader(data.coarse_path, data.coarse_pattern).load(identifiers)[0]
        fine_raster = GeoTiffLoader(data.fine_path, data.fine_pattern).load(identifiers)[0]
        coarse = coarse_raster.values
        fine = fine_raster.values
        self._reference_profile = fine_raster.profile
        label = None
        if data.label_path is not None:
            label = GeoTiffLoader(data.label_path, data.label_pattern).load(identifiers)[0].values
        return coarse, fine, label

    def run(self) -> dict[str, MethodResult]:
        coarse, fine, label = self.load_first_dataset()
        bands = self.config.band_count
        if label is not None and (label.ndim != 3 or label.shape[2] < bands):
            raise ValueError(f"标签影像至少需要 {bands} 个波段，实际形状为 {label.shape}。")

        results: dict[str, MethodResult] = {}
        if "dsck" in self.config.methods:
            started = perf_counter()
            prediction = DSCKInterpolator(
                self.config.dsck, self.config.search, self.config.backend.mode,
            ).sharpen(
                coarse, fine, bands, self._print_progress
            )
            results["dsck"] = MethodResult(
                prediction=prediction,
                metrics=self._evaluate(label, prediction, self.config.dsck.coarse_scale),
                elapsed_seconds=perf_counter() - started,
            )

        if "atprk" in self.config.methods:
            started = perf_counter()
            output = ATPRKInterpolator(
                self.config.atprk, self.config.search, self.config.backend.mode,
            ).sharpen(
                coarse, fine, bands, self._print_progress
            )
            spatial_scale = fine.shape[0] / coarse.shape[0]
            results["atprk"] = MethodResult(
                prediction=output.prediction,
                uncertainty=output.uncertainty,
                metrics=self._evaluate(label, output.prediction, spatial_scale),
                elapsed_seconds=perf_counter() - started,
            )
        return results

    def save_results(self, results: dict[str, MethodResult]) -> None:
        """Save predictions and optional uncertainty arrays as GeoTIFF files."""
        output = self.config.output
        if self._reference_profile is None:
            raise RuntimeError("尚未加载输入影像，无法确定输出 GeoTIFF 的空间参考。")
        output.directory.mkdir(parents=True, exist_ok=True)
        identifier = self.config.data.dates[0]
        for method, result in results.items():
            if output.save_prediction:
                save_geotiff(
                    output.directory / f"{identifier}_{method}.tif",
                    result.prediction,
                    self._reference_profile,
                )
            if output.save_uncertainty and result.uncertainty is not None:
                save_geotiff(
                    output.directory / f"{identifier}_{method}_uncertainty.tif",
                    result.uncertainty,
                    self._reference_profile,
                )

    def _evaluate(
        self,
        label: np.ndarray | None,
        prediction: np.ndarray,
        scale: float,
    ) -> SpectralMetrics | None:
        if self.config.mode == "real" or label is None:
            return None
        return evaluate_spectral(label[:, :, :self.config.band_count], prediction, scale)

    @staticmethod
    def _print_progress(method: str, current: int, total: int) -> None:
        print(f"{method} 进度：{current}/{total}")
