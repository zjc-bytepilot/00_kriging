"""GeoTIFF input and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio


@dataclass(frozen=True)
class RasterCube:
    """Image cube in HWC order together with its Rasterio profile."""

    values: np.ndarray
    profile: dict[str, Any]
    path: Path


class GeoTiffLoader:
    """Load GeoTIFF files and convert Rasterio CHW arrays to algorithm HWC arrays."""

    def __init__(
        self,
        directory: str | Path,
        file_pattern: str = "{identifier}.tif",
    ) -> None:
        self.directory = Path(directory)
        self.file_pattern = file_pattern
        if "{identifier}" not in file_pattern:
            raise ValueError("file_pattern 必须包含 {identifier} 占位符。")

    def load(self, identifiers: Iterable[str]) -> list[RasterCube]:
        cubes: list[RasterCube] = []
        for identifier in identifiers:
            file_path = self.directory / self.file_pattern.format(identifier=identifier)
            if not file_path.is_file():
                raise FileNotFoundError(f"GeoTIFF 文件不存在: {file_path}")
            with rasterio.open(file_path) as source:
                values = np.moveaxis(source.read(out_dtype="float32"), 0, -1)
                profile = source.profile.copy()
            if values.ndim != 3 or values.shape[2] == 0:
                raise ValueError(f"{file_path} 中没有有效波段，实际形状为 {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{file_path} 包含 NaN 或无穷值，请先处理 NoData 区域。")
            cubes.append(RasterCube(values=values, profile=profile, path=file_path))
        return cubes


def save_geotiff(
    path: str | Path,
    values: np.ndarray,
    reference_profile: Mapping[str, Any],
) -> None:
    """Save an HWC cube as GeoTIFF using a reference raster's spatial metadata."""
    if values.ndim != 3:
        raise ValueError(f"输出影像必须是三维 HWC 数组，实际形状为 {values.shape}。")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(reference_profile)
    profile.update(
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=values.shape[2],
        dtype="float32",
        compress="deflate",
    )
    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(np.moveaxis(values.astype(np.float32, copy=False), -1, 0))
