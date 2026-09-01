"""Benchmark pre-warmed ATPRK and DSCK calls on deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

from kriging.config import ATPRKConfig, DSCKConfig, SearchConfig
from kriging.estimators import ATPRKInterpolator, DSCKInterpolator
from kriging.spatial import downsample_plane, gaussian_psf


def synthetic_pair(size: int = 15, scale: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Create the same deterministic correlated pair used by smoke tests."""
    generator = np.random.default_rng(42)
    fine_plane = gaussian_filter(generator.normal(size=(size, size)), sigma=1.2)
    fine_plane = (fine_plane - fine_plane.min()) / (fine_plane.max() - fine_plane.min())
    psf = gaussian_psf(scale=scale, window=1, sigma=1.0)
    coarse_plane = downsample_plane(fine_plane, scale=scale, window=1, psf=psf)
    return coarse_plane[:, :, None], fine_plane[:, :, None]


def build_runner(method: str) -> Callable[[], object]:
    """Build one configured public API call for the selected method."""
    coarse, fine = synthetic_pair()
    search = SearchConfig(
        constant_min=0.5,
        sill_min=0.5,
        range_min=0.5,
        sill_steps=1,
        range_steps=1,
        constant_steps=1,
        step_size=0.1,
        max_lag=1,
    )
    if method == "atprk":
        model = ATPRKInterpolator(ATPRKConfig(window=1, psf_sigma=1.0), search)
        return lambda: model.sharpen(coarse, fine, band_count=1)
    model = DSCKInterpolator(
        DSCKConfig(
            coarse_scale=3,
            fine_scale=2,
            coarse_window=1,
            fine_window=1,
            psf_sigma=1.0,
        ),
        search,
    )
    return lambda: model.sharpen(coarse, fine, band_count=1)


def benchmark(method: str, repeats: int) -> dict[str, dict[str, float | int]]:
    """Warm a runner once, then return its mean pre-warmed execution time."""
    runner = build_runner(method)
    runner()
    durations: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        runner()
        durations.append(perf_counter() - started)
    return {method: {"seconds_per_run": float(np.mean(durations)), "runs": repeats}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("atprk", "dsck", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats 必须至少为 1")
    return args


def main() -> None:
    args = parse_args()
    methods = ("atprk", "dsck") if args.method == "both" else (args.method,)
    result: dict[str, dict[str, float | int]] = {}
    for method in methods:
        result.update(benchmark(method, args.repeats))
    json.dump(result, fp=sys.stdout)


if __name__ == "__main__":
    main()
