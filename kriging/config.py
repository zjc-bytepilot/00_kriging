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
class CloudMaskConfig:
    """一个 C-DSCK 云掩膜输入。"""

    name: str
    path: Path
    pattern: str = "M{identifier}.tif"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CloudMaskConfig":
        name = str(values.get("name", "")).strip()
        if not name:
            raise ValueError("data.cloud_masks 的每项都必须配置非空 name。")
        if "/" in name or "\\" in name:
            raise ValueError("data.cloud_masks 的 name 不能包含路径分隔符。")
        if not values.get("path"):
            raise ValueError(f"云掩膜 {name!r} 必须配置 path。")
        pattern = str(values.get("pattern", "M{identifier}.tif"))
        if "{identifier}" not in pattern:
            raise ValueError(f"云掩膜 {name!r} 的 pattern 必须包含 {{identifier}}。")
        return cls(name=name, path=_resolve_project_path(values["path"]), pattern=pattern)


@dataclass(frozen=True)
class DataConfig:
    coarse_path: Path
    fine_path: Path
    dates: tuple[str, ...]
    label_path: Path | None = None
    coarse_pattern: str = "C{identifier}.tif"
    fine_pattern: str = "F{identifier}.tif"
    label_pattern: str = "L{identifier}.tif"
    cloud_masks: tuple[CloudMaskConfig, ...] = ()

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DataConfig":
        dates = tuple(str(value) for value in values["dates"])
        if not dates:
            raise ValueError("data.dates 至少需要包含一个日期或文件标识。")
        raw_cloud_masks = values.get("cloud_masks", ())
        if isinstance(raw_cloud_masks, (str, bytes)) or not isinstance(raw_cloud_masks, (list, tuple)):
            raise ValueError("data.cloud_masks 必须是列表或元组。")
        cloud_masks = tuple(CloudMaskConfig.from_dict(mask) for mask in raw_cloud_masks)
        names = [mask.name for mask in cloud_masks]
        if len(names) != len(set(names)):
            raise ValueError("data.cloud_masks 中的 name 不能重复。")
        return cls(
            coarse_path=_resolve_project_path(values["coarse_path"]),
            fine_path=_resolve_project_path(values["fine_path"]),
            label_path=_resolve_project_path(values["label_path"]) if values.get("label_path") else None,
            dates=dates,
            coarse_pattern=values.get("coarse_pattern", "C{identifier}.tif"),
            fine_pattern=values.get("fine_pattern", "F{identifier}.tif"),
            label_pattern=values.get("label_pattern", "L{identifier}.tif"),
            cloud_masks=cloud_masks,
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
    """DSCK 配置。

    尺度参数分为两组,语义不同:

      - 经验阶段(插值/退化/经验变异函数/反卷积): ``coarse_scale`` 为
        coarse 到 fine 的倍数,``fine_scale`` 为 fine 到点尺度的倍数
        (点尺度 = fine / fine_scale);
      - 克里金矩阵阶段(r_* 支撑矩阵与 RHS): ``matrix_coarse_scale``
        (s0)与 ``matrix_fine_scale``(s),为 None 时回退到经验阶段取值。

    约束:``matrix_coarse_scale`` 必须等于 ``coarse_scale``,因为输出网格
    尺寸由 coarse × coarse_scale 决定,矩阵的子像元划分必须与之一致;
    ``matrix_fine_scale`` 可独立设置(只影响 fine 支撑协方差的数值)。
    """

    coarse_scale: int = 3
    fine_scale: int = 2
    coarse_window: int = 1
    fine_window: int = 3
    psf_sigma: float = 1.0
    cross_mode: str = "degrade"
    matrix_coarse_scale: int | None = None
    matrix_fine_scale: int | None = None

    def __post_init__(self) -> None:
        if min(self.coarse_scale, self.fine_scale, self.coarse_window, self.fine_window) <= 0:
            raise ValueError("DSCK 的尺度和窗口参数必须为正整数。")
        if self.psf_sigma <= 0:
            raise ValueError("psf_sigma 必须大于 0。")
        if self.cross_mode not in {"degrade", "interpolate"}:
            raise ValueError("cross_mode 只能是 'degrade' 或 'interpolate'。")
        if self.matrix_coarse_scale is not None and self.matrix_coarse_scale != self.coarse_scale:
            raise ValueError(
                "matrix_coarse_scale 必须等于 coarse_scale:"
                "输出网格由 coarse×coarse_scale 决定,矩阵子像元划分必须一致。"
            )
        if self.matrix_fine_scale is not None and self.matrix_fine_scale <= 0:
            raise ValueError("matrix_fine_scale 必须为正整数。")


@dataclass(frozen=True)
class ATPRKConfig:
    window: int = 1
    psf_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.window <= 0 or self.psf_sigma <= 0:
            raise ValueError("ATPRK 的 window 和 psf_sigma 必须大于 0。")


@dataclass(frozen=True)
class ATPKConfig:
    window: int = 1
    psf_sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.window <= 0 or self.psf_sigma <= 0:
            raise ValueError("ATPK 的 window 和 psf_sigma 必须大于 0。")


@dataclass(frozen=True)
class CDSCKConfig:
    """云感知 DSCK 配置。

    尺度语义(统一点尺度坐标系):
      - ``coarse_scale``:coarse 到 fine 的倍数(s0);
      - ``fine_scale``:fine 到点尺度的倍数(s),点尺度 = fine / fine_scale;
      - 一个 coarse 像元 = coarse_scale * fine_scale 个点尺度单位。

    ``cross_mode`` 决定交叉半方差的配对策略,与 DSCK 一致:
      - ``"interpolate"``(默认):ATPK 把 coarse 插值到 fine 尺度,交叉
        半方差在 fine 尺度下计算;
      - ``"degrade"``:把 Fine 退化到 coarse 尺度,交叉半方差在 coarse
        尺度下计算。
    两种模式只改变交叉经验变异函数的观测尺度,点尺度模型相同。预测时
    按 coarse 局部窗口共享候选点动态求解。

    ``matrix_coarse_scale``/``matrix_fine_scale`` 为克里金矩阵阶段的
    s0/s,为 None 时回退到经验阶段取值;``matrix_coarse_scale`` 必须等于
    ``coarse_scale``(输出网格约束),``matrix_fine_scale`` 可独立设置。
    """

    coarse_scale: int = 3
    fine_scale: int = 2
    coarse_window: int = 1
    fine_window: int = 3
    psf_sigma: float = 1.0
    cross_mode: str = "interpolate"
    matrix_coarse_scale: int | None = None
    matrix_fine_scale: int | None = None
    max_points: int = 100
    max_radius: int = 50
    batch_size: int = 512

    def __post_init__(self) -> None:
        if min(self.coarse_scale, self.fine_scale, self.coarse_window, self.fine_window) <= 0:
            raise ValueError("CDSCK 的尺度和窗口参数必须为正整数。")
        if self.psf_sigma <= 0:
            raise ValueError("psf_sigma 必须大于 0。")
        if self.cross_mode not in {"degrade", "interpolate"}:
            raise ValueError("CDSCK 的 cross_mode 只能是 'degrade' 或 'interpolate'。")
        if self.matrix_coarse_scale is not None and self.matrix_coarse_scale != self.coarse_scale:
            raise ValueError(
                "matrix_coarse_scale 必须等于 coarse_scale:"
                "输出网格由 coarse×coarse_scale 决定,矩阵子像元划分必须一致。"
            )
        if self.matrix_fine_scale is not None and self.matrix_fine_scale <= 0:
            raise ValueError("matrix_fine_scale 必须为正整数。")
        if self.max_points <= 0:
            raise ValueError("max_points 必须大于 0。")
        if self.max_radius <= 0:
            raise ValueError("max_radius 必须大于 0。")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")


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
    atpk: ATPKConfig = field(default_factory=ATPKConfig)
    atprk: ATPRKConfig = field(default_factory=ATPRKConfig)
    cdsck: CDSCKConfig = field(default_factory=CDSCKConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    mode: str = "degraded"
    methods: tuple[str, ...] = ("dsck", "atprk")
    band_count: int = 4

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExperimentConfig":
        methods = tuple(method.lower() for method in values.get("methods", ("dsck", "atprk")))
        unsupported = set(methods) - {"dsck", "atpk", "atprk", "c_dsck"}
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
        if "c_dsck" in methods and not data.cloud_masks:
            raise ValueError("c_dsck 方法必须配置至少一个 data.cloud_masks 项。")
        return cls(
            data=data,
            search=SearchConfig(**values.get("search", {})),
            dsck=DSCKConfig(**values.get("dsck", {})),
            atpk=ATPKConfig(**values.get("atpk", {})),
            atprk=ATPRKConfig(**values.get("atprk", {})),
            cdsck=CDSCKConfig(**values.get("cdsck", {})),
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
