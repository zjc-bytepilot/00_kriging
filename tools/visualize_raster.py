"""Visualize a GeoTIFF raster with sensible NoData handling.

The command is generic, while presets provide convenient defaults for common
products such as SMAP soil moisture.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VisualizationStyle:
    cmap: str
    colorbar_label: str
    value_min: float | None = None
    value_max: float | None = None


PRESETS = {
    "generic": VisualizationStyle("viridis", "Value"),
    "soil_moisture": VisualizationStyle(
        "YlGnBu",
        "Volumetric soil moisture (m³/m³)",
        0.0,
        0.6,
    ),
    "smp": VisualizationStyle(
        "YlGnBu",
        "Volumetric soil moisture (m³/m³)",
        0.0,
        0.6,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化单波段 GeoTIFF 数据")
    parser.add_argument("input", type=Path, help="输入 GeoTIFF 文件")
    parser.add_argument("--band", type=int, default=1, help="波段编号，从 1 开始")
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="generic",
        help="显示预设；SMP 土壤湿度可选 smp 或 soil_moisture",
    )
    parser.add_argument("--cmap", help="覆盖预设的 Matplotlib 颜色表")
    parser.add_argument("--vmin", type=float, help="颜色条最小值")
    parser.add_argument("--vmax", type=float, help="颜色条最大值")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(2.0, 98.0),
        help="generic 模式的显示百分位范围（默认：2 98）",
    )
    parser.add_argument("--title", help="图片标题；默认使用文件名")
    parser.add_argument("--colorbar-label", help="覆盖颜色条标签")
    parser.add_argument("--dpi", type=int, default=180, help="输出分辨率")
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 PNG；默认保存到 tmp/visualizations/<文件名>.png",
    )
    parser.add_argument("--show", action="store_true", help="保存后打开交互窗口")
    return parser.parse_args()


def _display_limits(
    values: np.ma.MaskedArray,
    style: VisualizationStyle,
    requested_min: float | None,
    requested_max: float | None,
    percentiles: tuple[float, float],
) -> tuple[float, float]:
    valid = values.compressed()
    if valid.size == 0:
        raise ValueError("影像没有有效像元。")
    low, high = percentiles
    if not 0 <= low < high <= 100:
        raise ValueError("percentiles 必须满足 0 <= LOW < HIGH <= 100。")
    value_min = requested_min if requested_min is not None else style.value_min
    value_max = requested_max if requested_max is not None else style.value_max
    if value_min is None:
        value_min = float(np.percentile(valid, low))
    if value_max is None:
        value_max = float(np.percentile(valid, high))
    if value_min >= value_max:
        raise ValueError("显示最小值必须小于最大值。")
    return value_min, value_max


def visualize(
    input_path: Path,
    output_path: Path,
    band: int = 1,
    preset: str = "generic",
    cmap: str | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    title: str | None = None,
    colorbar_label: str | None = None,
    dpi: int = 180,
    show: bool = False,
) -> dict[str, float | int | str]:
    """Render one raster band and return summary statistics."""
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    style = PRESETS[preset]
    with rasterio.open(input_path) as source:
        if not 1 <= band <= source.count:
            raise ValueError(f"band 应位于 1 到 {source.count} 之间。")
        values = source.read(band, masked=True).astype(np.float64)
        values = np.ma.masked_invalid(values)
        bounds = source.bounds
        crs = source.crs

    valid = values.compressed()
    display_min, display_max = _display_limits(
        values, style, value_min, value_max, percentiles
    )

    matplotlib_cache = PROJECT_ROOT / "tmp" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    if not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    image = axis.imshow(
        values,
        cmap=cmap or style.cmap,
        vmin=display_min,
        vmax=display_max,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        interpolation="nearest",
    )
    axis.set_title(title or input_path.stem)
    axis.set_xlabel("X" if crs is None else f"X ({crs.to_string()})")
    axis.set_ylabel("Y")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label(colorbar_label or style.colorbar_label)
    stats_text = (
        f"valid={valid.size:,}  min={valid.min():.4g}  "
        f"mean={valid.mean():.4g}  max={valid.max():.4g}"
    )
    axis.text(
        0.01,
        0.01,
        stats_text,
        transform=axis.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return {
        "valid_pixels": int(valid.size),
        "minimum": float(valid.min()),
        "mean": float(valid.mean()),
        "maximum": float(valid.max()),
        "display_min": display_min,
        "display_max": display_max,
        "output": str(output_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output or (
        PROJECT_ROOT / "tmp" / "visualizations" / f"{input_path.stem}.png"
    )
    stats = visualize(
        input_path=input_path,
        output_path=output_path,
        band=args.band,
        preset=args.preset,
        cmap=args.cmap,
        value_min=args.vmin,
        value_max=args.vmax,
        percentiles=tuple(args.percentiles),
        title=args.title,
        colorbar_label=args.colorbar_label,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"有效像元：{stats['valid_pixels']:,}")
    print(
        f"数值范围：{stats['minimum']:.6g} ~ {stats['maximum']:.6g}，"
        f"均值：{stats['mean']:.6g}"
    )
    print(f"图片已保存：{stats['output']}")


if __name__ == "__main__":
    main()
