"""Discover and pair GF6/Landsat GeoTIFFs by their numeric serial."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SERIAL_PATTERN = re.compile(r"^(?P<serial>\d+)_")


@dataclass(frozen=True)
class DegradedPair:
    serial: str
    gf6_path: Path
    landsat_path: Path


def _index_by_serial(directory: Path) -> dict[str, Path]:
    """Index GeoTIFFs in *directory*, rejecting invalid and duplicate names."""
    indexed: dict[str, Path] = {}
    for path in sorted(directory.glob("*.tif")):
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
