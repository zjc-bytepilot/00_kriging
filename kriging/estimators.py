"""Object-oriented interfaces around the numerical kriging kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import atprk, dsck
from .spatial import gaussian_psf
from .config import ATPRKConfig, DSCKConfig, SearchConfig

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ATPRKResult:
    prediction: np.ndarray
    uncertainty: np.ndarray


def _validate_cubes(coarse: np.ndarray, fine: np.ndarray, band_count: int | None) -> int:
    if coarse.ndim != 3 or fine.ndim != 3:
        raise ValueError("coarse 和 fine 必须是 (高度, 宽度, 波段) 三维数组。")
    if coarse.shape[2] != fine.shape[2]:
        raise ValueError("coarse 和 fine 的波段数必须一致。")
    selected_bands = coarse.shape[2] if band_count is None else band_count
    if not 0 < selected_bands <= coarse.shape[2]:
        raise ValueError(f"band_count 应位于 1 到 {coarse.shape[2]} 之间。")
    return selected_bands


class DSCKInterpolator:
    """Dual-support cokriging downscaler configured once and reused per image."""

    def __init__(self, config: DSCKConfig, search: SearchConfig) -> None:
        self.config = config
        self.search = search
        self._coarse_psf = gaussian_psf(config.coarse_scale, config.coarse_window, config.psf_sigma)
        self._fine_psf = gaussian_psf(config.fine_scale, config.fine_window, config.psf_sigma)

    def sharpen_band(self, coarse: np.ndarray, fine: np.ndarray) -> np.ndarray:
        cfg, search = self.config, self.search
        return dsck.DSCK_Regression_Sharpen(
            coarse, fine,
            search.constant_min, search.sill_min, search.range_min,
            search.sill_steps, search.range_steps, search.constant_steps,
            search.step_size, search.max_lag,
            cfg.coarse_window, cfg.fine_window,
            self._coarse_psf, self._fine_psf,
            cfg.coarse_scale, cfg.fine_scale,
        )

    def sharpen(
        self,
        coarse: np.ndarray,
        fine: np.ndarray,
        band_count: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        bands = _validate_cubes(coarse, fine, band_count)
        expected_shape = (coarse.shape[0] * self.config.coarse_scale,
                          coarse.shape[1] * self.config.coarse_scale)
        if fine.shape[:2] != expected_shape:
            raise ValueError(
                f"DSCK fine 空间形状应为 {expected_shape}，实际为 {fine.shape[:2]}。"
            )
        prediction = np.empty((*expected_shape, bands), dtype=np.result_type(coarse, fine))
        for band in range(bands):
            prediction[:, :, band] = self.sharpen_band(coarse[:, :, band], fine[:, :, band])
            if progress:
                progress("DSCK", band + 1, bands)
        return prediction


class ATPRKInterpolator:
    """Area-to-point regression kriging downscaler."""

    def __init__(self, config: ATPRKConfig, search: SearchConfig) -> None:
        self.config = config
        self.search = search

    def sharpen_band(self, coarse: np.ndarray, fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if coarse.ndim != 2 or fine.ndim != 2:
            raise ValueError("单波段 coarse 和 fine 必须是二维数组。")
        if fine.shape[0] % coarse.shape[0] or fine.shape[1] % coarse.shape[1]:
            raise ValueError("fine 的空间尺寸必须是 coarse 的整数倍。")
        row_scale = fine.shape[0] // coarse.shape[0]
        column_scale = fine.shape[1] // coarse.shape[1]
        if row_scale != column_scale:
            raise ValueError("ATPRK 仅支持行列缩放比例相同的数据。")
        psf = gaussian_psf(row_scale, self.config.window, self.config.psf_sigma)
        search = self.search
        return atprk.ATPRK_Sharpen(
            coarse, fine,
            search.sill_min, search.range_min,
            search.sill_steps, search.range_steps,
            search.step_size, search.max_lag,
            self.config.window, psf,
        )

    def sharpen(
        self,
        coarse: np.ndarray,
        fine: np.ndarray,
        band_count: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> ATPRKResult:
        bands = _validate_cubes(coarse, fine, band_count)
        prediction = np.empty((*fine.shape[:2], bands), dtype=np.result_type(coarse, fine))
        uncertainty = np.empty_like(prediction)
        for band in range(bands):
            uncertainty[:, :, band], prediction[:, :, band] = self.sharpen_band(
                coarse[:, :, band], fine[:, :, band]
            )
            if progress:
                progress("ATPRK", band + 1, bands)
        return ATPRKResult(prediction=prediction, uncertainty=uncertainty)
