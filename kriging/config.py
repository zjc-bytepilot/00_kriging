"""Typed configuration objects loaded from Python ``CONFIG`` dictionaries."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str | Path) -> Path:
    """Resolve configuration paths relative to the project root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class DataConfig:
    coarse_path: Path
    fine_path: Path
    dates: tuple[str, ...]
    label_path: Path | None = None
    coarse_pattern: str = "C{identifier}.tif"
    fine_pattern: str = "F{identifier}.tif"
    label_pattern: str = "L{identifier}.tif"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DataConfig":
        dates = tuple(str(value) for value in values["dates"])
        if not dates:
            raise ValueError("data.dates 至少需要包含一个日期或文件标识。")
        return cls(
            coarse_path=_resolve_project_path(values["coarse_path"]),
            fine_path=_resolve_project_path(values["fine_path"]),
            label_path=_resolve_project_path(values["label_path"]) if values.get("label_path") else None,
            dates=dates,
            coarse_pattern=values.get("coarse_pattern", "C{identifier}.tif"),
            fine_pattern=values.get("fine_pattern", "F{identifier}.tif"),
            label_pattern=values.get("label_pattern", "L{identifier}.tif"),
        )


@dataclass(frozen=True)
class SearchConfig:
    constant_min: float = 0.5
    sill_min: float = 0.5
    range_min: float = 0.5
    sill_steps: int = 30
    range_steps: int = 30
    constant_steps: int = 30
    step_size: float = 0.1
    max_lag: int = 30

    def __post_init__(self) -> None:
        if min(self.sill_steps, self.range_steps, self.constant_steps, self.max_lag) <= 0:
            raise ValueError("搜索步数和 max_lag 必须为正整数。")
        if self.step_size <= 0:
            raise ValueError("step_size 必须大于 0。")


@dataclass(frozen=True)
class DSCKConfig:
    coarse_scale: int = 3
    fine_scale: int = 2
    coarse_window: int = 1
    fine_window: int = 3
    psf_sigma: float = 1.0
    cross_mode: str = "degrade"

    def __post_init__(self) -> None:
        if min(self.coarse_scale, self.fine_scale, self.coarse_window, self.fine_window) <= 0:
            raise ValueError("DSCK 的尺度和窗口参数必须为正整数。")
        if self.psf_sigma <= 0:
            raise ValueError("psf_sigma 必须大于 0。")
        if self.cross_mode not in {"degrade", "interpolate"}:
            raise ValueError("cross_mode 只能是 'degrade' 或 'interpolate'。")


@dataclass(frozen=True)
class ATPRKConfig:
    window: int = 1
    psf_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.window <= 0 or self.psf_sigma <= 0:
            raise ValueError("ATPRK 的 window 和 psf_sigma 必须大于 0。")


@dataclass(frozen=True)
class OutputConfig:
    directory: Path = Path("tmp/results")
    save_prediction: bool = True
    save_uncertainty: bool = True

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OutputConfig":
        return cls(
            directory=_resolve_project_path(values.get("directory", "tmp/results")),
            save_prediction=bool(values.get("save_prediction", True)),
            save_uncertainty=bool(values.get("save_uncertainty", True)),
        )


@dataclass(frozen=True)
class BackendConfig:
    """Execution backend selection for the numerical kernels."""

    mode: str = "cpu"

    def __post_init__(self) -> None:
        if self.mode not in {"cpu", "gpu", "auto"}:
            raise ValueError("backend.mode 只能是 'cpu'、'gpu' 或 'auto'。")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    search: SearchConfig = field(default_factory=SearchConfig)
    dsck: DSCKConfig = field(default_factory=DSCKConfig)
    atprk: ATPRKConfig = field(default_factory=ATPRKConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    mode: str = "degraded"
    methods: tuple[str, ...] = ("dsck", "atprk")
    band_count: int = 4

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        methods = tuple(method.lower() for method in values.get("methods", ("dsck", "atprk")))
        unsupported = set(methods) - {"dsck", "atprk"}
        if unsupported:
            raise ValueError(f"不支持的算法: {sorted(unsupported)}")
        band_count = int(values.get("band_count", 4))
        if band_count <= 0:
            raise ValueError("band_count 必须为正整数。")
        mode = str(values.get("mode", "degraded")).lower()
        if mode not in {"degraded", "real"}:
            raise ValueError("mode 只能是 'degraded' 或 'real'。")
        data = DataConfig.from_dict(values["data"])
        if mode == "degraded" and data.label_path is None:
            raise ValueError("degraded 模式必须配置 data.label_path。")
        return cls(
            data=data,
            search=SearchConfig(**values.get("search", {})),
            dsck=DSCKConfig(**values.get("dsck", {})),
            atprk=ATPRKConfig(**values.get("atprk", {})),
            output=OutputConfig.from_dict(values.get("output", {})),
            backend=BackendConfig(**values.get("backend", {})),
            mode=mode,
            methods=methods,
            band_count=band_count,
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate ``CONFIG`` from a local Python configuration file."""
    config_path = Path(path)
    if config_path.suffix.lower() != ".py":
        raise ValueError("配置文件必须是 .py 文件。")
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    spec = importlib.util.spec_from_file_location("_kriging_user_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = getattr(module, "CONFIG", None)
    if not isinstance(values, Mapping):
        raise TypeError("配置文件必须定义名为 CONFIG 的字典。")
    return ExperimentConfig.from_dict(values)
