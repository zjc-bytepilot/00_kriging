"""Command-line entry point for kriging downscaling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from kriging import load_config
from kriging.pipeline import KrigingExperiment, MethodResult

PROJECT_ROOT = Path(__file__).resolve().parent


def print_result(method: str, result: MethodResult) -> None:
    metrics = result.metrics
    if metrics is None:
        print(f"{method.upper()} 已完成（真实数据模式，不计算参考指标）")
        print(f"耗时：{result.elapsed_seconds:.2f} 秒")
        return
    print(f"{method.upper()} 精度评定")
    print(f"RMSE：{metrics.rmse}")
    print(f"CC：{metrics.correlation}")
    print(f"ERGAS：{metrics.ergas}")
    print(f"UIQI：{metrics.uiqi}")
    print(f"SAM：{metrics.sam}")
    print(f"耗时：{result.elapsed_seconds:.2f} 秒")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 DSCK/ATPRK 克里金降尺度实验")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "degraded.py",
        help="Python 配置文件路径（默认：config/degraded.py）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = KrigingExperiment(load_config(args.config))
    results = experiment.run()
    experiment.save_results(results)
    for method, result in results.items():
        print_result(method, result)
    print(f"结果已保存到：{experiment.config.output.directory.resolve()}")


if __name__ == "__main__":
    main()
