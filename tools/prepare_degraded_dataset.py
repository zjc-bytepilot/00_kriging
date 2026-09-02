"""Discover and pair GF6/Landsat GeoTIFFs by their numeric serial."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import rasterio
from affine import Affine

from kriging.spatial import downsample_plane, gaussian_psf


SERIAL_PATTERN = re.compile(r"^(?P<serial>\d+)_")


@dataclass(frozen=True)
class DegradedPair:
    serial: str
    gf6_path: Path
    landsat_path: Path


def _index_by_serial(directory: Path) -> dict[str, Path]:
    """Index GeoTIFFs in *directory*, rejecting invalid and duplicate names."""
    indexed: dict[str, Path] = {}
    for path in sorted(path for path in directory.glob("*.tif") if path.is_file()):
        match = SERIAL_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"文件命名规则无效：{path.name}")
        serial = match.group("serial")
        if serial in indexed:
            raise ValueError(f"发现重复序号：{serial}")
        indexed[serial] = path
    return indexed


class DegradedPairDataset:
    """A lightweight collection of serial-paired GF6 and Landsat files."""

    def __init__(self, input_root: str | Path) -> None:
        root = Path(input_root)
        gf6 = _index_by_serial(root / "gf6")
        landsat = _index_by_serial(root / "landsat")
        if gf6.keys() != landsat.keys():
            raise ValueError("GF6 与 Landsat 存在缺少配对的序号。")
        self.pairs = tuple(
            DegradedPair(serial, gf6[serial], landsat[serial])
            for serial in sorted(gf6, key=int)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[DegradedPair]:
        return iter(self.pairs)


def _downsample_cube(
    values: np.ndarray, *, scale: int, window: int, psf: np.ndarray
) -> np.ndarray:
    """Downsample each spectral band independently."""
    return np.stack(
        [
            downsample_plane(values[:, :, band], scale, window, psf)
            for band in range(values.shape[2])
        ],
        axis=-1,
    )


def _scaled_profile(
    profile: Mapping[str, Any], values: np.ndarray, scale: int
) -> dict[str, Any]:
    """Return a GeoTIFF profile for values reduced by an integer scale."""
    result = dict(profile)
    result.update(
        height=values.shape[0],
        width=values.shape[1],
        transform=profile["transform"] @ Affine.scale(scale),
    )
    return result


class DegradationProcessor:
    """Generate the fine, coarse, and label GeoTIFF triplet for one pair."""

    def __init__(self, scale: int = 3, window: int = 1, psf_sigma: float = 1.0) -> None:
        self.scale = scale
        self.window = window
        self.psf_sigma = psf_sigma
        self.psf = gaussian_psf(scale, window, psf_sigma)

    def process_pair(
        self,
        pair: DegradedPair,
        output_root: str | Path,
        overwrite: bool = False,
    ) -> dict[str, str]:
        """Write PSF-degraded GF6/Landsat rasters and the Landsat label."""
        with rasterio.open(pair.gf6_path) as gf6_dataset:
            gf6_profile = gf6_dataset.profile.copy()
            gf6_values = np.moveaxis(gf6_dataset.read(), 0, -1)
        with rasterio.open(pair.landsat_path) as landsat_dataset:
            landsat_profile = landsat_dataset.profile.copy()
            landsat_values = np.moveaxis(landsat_dataset.read(), 0, -1)

        self._validate_shapes(gf6_values, landsat_values)

        root = Path(output_root)
        targets = {
            "fine": root / "fine" / f"F{pair.serial}.tif",
            "coarse": root / "coarse" / f"C{pair.serial}.tif",
            "label": root / "label" / f"L{pair.serial}.tif",
        }
        if not overwrite:
            existing = next((path for path in targets.values() if path.exists()), None)
            if existing is not None:
                raise FileExistsError(f"输出文件已存在：{existing}")

        fine_values = _downsample_cube(
            gf6_values, scale=self.scale, window=self.window, psf=self.psf
        )
        coarse_values = _downsample_cube(
            landsat_values, scale=self.scale, window=self.window, psf=self.psf
        )
        profiles = {
            "fine": _scaled_profile(gf6_profile, fine_values, self.scale),
            "coarse": _scaled_profile(landsat_profile, coarse_values, self.scale),
            "label": landsat_profile,
        }
        values = {
            "fine": fine_values,
            "coarse": coarse_values,
            "label": landsat_values,
        }
        for name, path in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", **profiles[name]) as dataset:
                dataset.write(np.moveaxis(values[name], -1, 0))

        return {name: str(path) for name, path in targets.items()}

    def _validate_shapes(self, gf6_values: np.ndarray, landsat_values: np.ndarray) -> None:
        for name, values in (("GF6", gf6_values), ("Landsat", landsat_values)):
            if values.shape[0] % self.scale or values.shape[1] % self.scale:
                raise ValueError(f"{name} 的行列尺寸必须能被 scale 整除。")
        if gf6_values.shape[2] != landsat_values.shape[2]:
            raise ValueError("GF6 与 Landsat 的波段数必须相同。")
        if (
            gf6_values.shape[0] // self.scale,
            gf6_values.shape[1] // self.scale,
        ) != landsat_values.shape[:2]:
            raise ValueError("GF6 下采样后的形状必须与 Landsat 形状一致。")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for batch dataset preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.scale <= 0 or args.window <= 0 or args.psf_sigma <= 0:
        parser.error("--scale、--window 和 --psf-sigma 必须大于 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare every serial-paired source image and write its manifest."""
    args = parse_args(argv)
    dataset = DegradedPairDataset(args.input_root)
    processor = DegradationProcessor(args.scale, args.window, args.psf_sigma)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在：{manifest_path}")

    pairs: list[dict[str, str]] = []
    for pair in dataset:
        outputs = processor.process_pair(pair, args.output_root, overwrite=args.overwrite)
        pairs.append(
            {
                "serial": pair.serial,
                "gf6_source": str(pair.gf6_path),
                "landsat_source": str(pair.landsat_path),
                **outputs,
            }
        )

    manifest = {
        "scale": args.scale,
        "window": args.window,
        "psf_sigma": args.psf_sigma,
        "pairs": pairs,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    json.dump({"output_root": str(args.output_root), "pair_count": len(pairs)}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
