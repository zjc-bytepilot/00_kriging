"""Quality metrics for multi-band downscaling results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralMetrics:
    rmse: np.ndarray
    correlation: np.ndarray
    ergas: float
    uiqi: np.ndarray
    sam: float


def evaluate_spectral(reference: np.ndarray, prediction: np.ndarray, scale: float) -> SpectralMetrics:
    """Compute the legacy RMSE, CC, ERGAS, UIQI and SAM metrics."""
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError(
            "reference 和 prediction 必须是形状相同的三维数组，"
            f"实际为 {reference.shape} 和 {prediction.shape}。"
        )
    if scale <= 0:
        raise ValueError("scale 必须大于 0。")

    height, width, bands = reference.shape
    rmse_values: list[float] = []
    correlation_values: list[float] = []
    relative_errors: list[float] = []
    uiqi_values: list[float] = []

    for band in range(bands):
        predicted = prediction[:, :, band]
        actual = reference[:, :, band]
        rmse = float(np.sqrt(np.sum((actual - predicted) ** 2) / (height * width)))
        covariance = np.sum(predicted * actual) - height * width * np.mean(predicted) * np.mean(actual)
        predicted_variance = np.sum(predicted ** 2) - height * width * np.mean(predicted) ** 2
        actual_variance = np.sum(actual ** 2) - height * width * np.mean(actual) ** 2
        denominator = np.sqrt(predicted_variance * actual_variance)
        correlation = float(covariance / denominator) if denominator else float("nan")
        actual_mean = float(np.mean(actual))
        relative_errors.append(rmse / actual_mean if actual_mean else float("nan"))
        uiqi_denominator = (
            (np.mean(predicted) ** 2 + actual_mean ** 2)
            * (predicted_variance + actual_variance)
        )
        uiqi = float(4 * np.mean(predicted) * actual_mean * covariance / uiqi_denominator) \
            if uiqi_denominator else float("nan")
        rmse_values.append(rmse)
        correlation_values.append(correlation)
        uiqi_values.append(uiqi)

    dot_products = np.sum(prediction * reference, axis=2)
    norms = np.linalg.norm(prediction, axis=2) * np.linalg.norm(reference, axis=2)
    cosines = np.divide(dot_products, norms, out=np.zeros_like(dot_products, dtype=float), where=norms > 0)
    sam = float(np.arccos(np.clip(np.mean(cosines), -1.0, 1.0)))
    return SpectralMetrics(
        rmse=np.asarray(rmse_values + [float(np.mean(rmse_values))]),
        correlation=np.asarray(correlation_values + [float(np.mean(correlation_values))]),
        ergas=float(100 * np.linalg.norm(relative_errors) / (scale * np.sqrt(bands))),
        uiqi=np.asarray(uiqi_values + [float(np.mean(uiqi_values))]),
        sam=sam,
    )
