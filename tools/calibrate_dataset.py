"""批量把 DN 缩放整数的 SR 影像还原为 float32 地表反射率。

按输入根目录下的子目录名推断传感器（``gf6``/``landsat``/``sentinel``），
对每个 ``.tif`` 应用对应定标系数 ``scale * DN + offset``，写入输出根目录
的同名子目录。不重投影、不裁剪。

典型用法::

    python -m tools.calibrate_dataset \\
      --input-root data/lan_gf \\
      --output-root data/lan_gf_ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from kriging.calibration import (
    NODATA_VALUE,
    SENSOR_BY_DIR,
    Sensor,
    load_calibrate_save,
    sensor_for_directory,
)


def _discover_sources(input_root: Path) -> list[tuple[Path, Sensor]]:
    """扫描 input_root 的子目录，按目录名推断传感器并收集 .tif 文件。"""
    discovered: list[tuple[Path, Sensor]] = []
    for child in sorted(p for p in input_root.iterdir() if p.is_dir()):
        sensor = sensor_for_directory(child.name)
        for tif in sorted(child.glob("*.tif")):
            if tif.is_file():
                discovered.append((tif, sensor))
    if not discovered:
        raise ValueError(
            f"输入根目录下未发现可处理的 .tif 文件：{input_root}"
        )
    return discovered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.input_root.is_dir():
        parser.error(f"--input-root 不是有效目录：{args.input_root}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """批量定标并写 manifest。"""
    args = parse_args(argv)
    sources = _discover_sources(args.input_root)

    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在：{manifest_path}")

    entries: list[dict[str, Any]] = []
    for source_path, sensor in sources:
        relative = source_path.relative_to(args.input_root)
        destination_path = args.output_root / relative

        if not args.overwrite and destination_path.exists():
            raise FileExistsError(f"输出文件已存在：{destination_path}")

        entry = load_calibrate_save(source_path, destination_path, sensor)
        entries.append(entry)

    manifest = {
        "nodata": NODATA_VALUE,
        "sensor_by_dir": {d: s.value for d, s in SENSOR_BY_DIR.items()},
        "files": entries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    json.dump(
        {"output_root": str(args.output_root), "file_count": len(entries)},
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
