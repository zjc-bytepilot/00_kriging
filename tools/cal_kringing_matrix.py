"""Generate DSCK kriging matrices for configured training images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kriging import dsck
from kriging.spatial import gaussian_psf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from kriging.config import load_config
from kriging.data import GeoTiffLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算并保存 DSCK 半方差矩阵")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "matrix.py",
        help="实验配置文件（默认：config/matrix.py）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tmp" / "kriging_matrices",
        help="矩阵输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config.data
    dsck_config = config.dsck
    search = config.search

    coarse_images = [item.values for item in GeoTiffLoader(
        data.coarse_path, data.coarse_pattern
    ).load(data.dates)]
    fine_images = [item.values for item in GeoTiffLoader(
        data.fine_path, data.fine_pattern
    ).load(data.dates)]
    if len(coarse_images) != len(fine_images):
        raise ValueError("粗、细分辨率训练影像数量不一致。")

    psf1 = gaussian_psf(
        dsck_config.coarse_scale, dsck_config.coarse_window, dsck_config.psf_sigma
    )
    psf2 = gaussian_psf(
        dsck_config.fine_scale, dsck_config.fine_window, dsck_config.psf_sigma
    )
    matrix_left_items: list[torch.Tensor] = []
    matrix_right_items: list[torch.Tensor] = []
    total = len(coarse_images) * config.band_count

    for image_index, (coarse_image, fine_image) in enumerate(zip(coarse_images, fine_images)):
        if coarse_image.shape[2] < config.band_count or fine_image.shape[2] < config.band_count:
            raise ValueError(
                f"第 {image_index + 1} 组影像少于 {config.band_count} 个波段。"
            )
        for band in range(config.band_count):
            matrix_left, vectors = dsck.calculate_matrix(
                coarse_image[:, :, band],
                fine_image[:, :, band],
                search.constant_min,
                search.sill_min,
                search.range_min,
                search.sill_steps,
                search.range_steps,
                search.constant_steps,
                search.step_size,
                search.max_lag,
                dsck_config.coarse_window,
                dsck_config.fine_window,
                psf1,
                psf2,
                dsck_config.coarse_scale,
                dsck_config.fine_scale,
            )
            matrix_right = np.concatenate(vectors, axis=1)
            matrix_left_items.append(torch.as_tensor(matrix_left, dtype=torch.float32).unsqueeze(0))
            matrix_right_items.append(torch.as_tensor(matrix_right, dtype=torch.float32).unsqueeze(0))
            current = image_index * config.band_count + band + 1
            print(f"克里金半方差矩阵计算进度：{current}/{total}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(matrix_left_items, dim=0), args.output_dir / "matrix_left.pt")
    torch.save(torch.cat(matrix_right_items, dim=0), args.output_dir / "matrix_right.pt")
    print(f"矩阵已保存到：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
