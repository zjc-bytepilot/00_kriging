"""多传感器地表反射率定标。

数据已是经过辐射定标与大气校正的地表反射率（SR）产品，仅以 uint16 缩放
整数存储。本模块负责将其按公式还原为 float32 反射率：``scale * DN + offset``。

仅做线性缩放还原，不做辐射定标、不做大气校正、不做越界过滤。后续的降采样
由 ``prepare_degraded_dataset`` 负责。

nodata 约定
-----------
源影像以 ``nodata=0`` 表示无效像元。定标后输出统一使用哨兵值
``NODATA_VALUE = -9999``（遥感惯例），并在 profile 中写回 ``nodata=-9999``。
之所以不用 ``NaN``，是因为 ``kriging.data.GeoTiffLoader`` 会拒绝包含 NaN
的影像；哨兵值可让既有加载链路零改动。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import rasterio

from kriging.data import RasterCube

#: 定标输出统一使用的 nodata 哨兵值。
NODATA_VALUE: float = -9999.0


class Sensor(str, Enum):
    """支持的传感器及其地表反射率产品级别。"""

    LANDSAT_C2_L2 = "landsat_c2_l2"
    GF6_SR = "gf6_sr"
    SENTINEL2_L2A = "sentinel2_l2a"


@dataclass(frozen=True)
class ScaleOffset:
    """线性定标系数：``reflectance = DN * scale + offset``。"""

    scale: float
    offset: float = 0.0


#: 各传感器到定标系数的注册表。新增传感器只需在此登记。
REGISTRY: dict[Sensor, ScaleOffset] = {
    # Landsat Collection 2 Level-2 SR 官方公式：ρ = DN*2.75e-5 - 0.2
    Sensor.LANDSAT_C2_L2: ScaleOffset(2.75e-5, -0.2),
    # GF6 SR 产品惯例：反射率 = DN * 1e-4（数据文件未写系数，由数值范围反推）
    Sensor.GF6_SR: ScaleOffset(1e-4, 0.0),
    # 当前 SAFE 元数据为 BOA_ADD_OFFSET=-1000、BOA_QUANTIFICATION_VALUE=10000。
    Sensor.SENTINEL2_L2A: ScaleOffset(1e-4, -0.1),
}

#: 输入目录名到传感器的推断映射。CLI 按子目录名查此表。
SENSOR_BY_DIR: dict[str, Sensor] = {
    "gf6": Sensor.GF6_SR,
    "landsat": Sensor.LANDSAT_C2_L2,
    "sentinel": Sensor.SENTINEL2_L2A,
}


def sensor_for_directory(directory_name: str) -> Sensor:
    """按输入子目录名推断传感器，未知目录名抛错以防用错系数。"""
    try:
        return SENSOR_BY_DIR[directory_name]
    except KeyError as error:
        raise ValueError(
            f"无法按目录名推断传感器：{directory_name!r}。"
            f"已知目录名：{sorted(SENSOR_BY_DIR)}。"
        ) from error


def calibrate_cube(cube: RasterCube, sensor: Sensor) -> RasterCube:
    """把一个 DN 缩放整数立方体按公式还原为地表反射率。

    仅做 ``scale * DN + offset`` 线性转换，不做越界过滤、不做大气校正。
    源 ``nodata``（profile 声明的值，或默认的 0）掩为哨兵值，输出 float32
    并在 profile 写回 ``nodata=NODATA_VALUE``。保留输入的几何与波段结构，
    不重投影、不裁剪。后续的降采样由 ``prepare_degraded_dataset`` 负责。
    """
    coefficients = REGISTRY[sensor]
    source_nodata = cube.profile.get("nodata")

    values = cube.values.astype(np.float32, copy=True)

    # 源 nodata 掩膜：profile 声明的值，或默认的 0。
    if source_nodata is not None:
        invalid = values == np.float32(source_nodata)
    else:
        invalid = values == np.float32(0)

    calibrated = values * np.float32(coefficients.scale) + np.float32(coefficients.offset)

    calibrated[invalid] = np.float32(NODATA_VALUE)

    profile = dict(cube.profile)
    profile.update(dtype="float32", nodata=NODATA_VALUE)

    return RasterCube(values=calibrated, profile=profile, path=cube.path)


def load_calibrate_save(
    source_path: str | Path,
    destination_path: str | Path,
    sensor: Sensor,
) -> Mapping[str, Any]:
    """读单个 GeoTIFF，定标，写回，返回处理摘要。"""
    with rasterio.open(source_path) as dataset:
        values = np.moveaxis(dataset.read(out_dtype="float32"), 0, -1)
        profile = dataset.profile.copy()
    if values.ndim != 3 or values.shape[2] == 0:
        raise ValueError(
            f"{source_path} 中没有有效波段，实际形状为 {values.shape}"
        )

    cube = RasterCube(values=values, profile=profile, path=Path(source_path))
    calibrated = calibrate_cube(cube, sensor)

    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out_profile = dict(calibrated.profile)
    out_profile.update(
        driver="GTiff",
        height=calibrated.values.shape[0],
        width=calibrated.values.shape[1],
        count=calibrated.values.shape[2],
        compress="deflate",
    )
    with rasterio.open(destination, "w", **out_profile) as destination_dataset:
        destination_dataset.write(np.moveaxis(calibrated.values, -1, 0))

    coefficients = REGISTRY[sensor]
    return {
        "source": str(source_path),
        "destination": str(destination),
        "sensor": sensor.value,
        "scale": coefficients.scale,
        "offset": coefficients.offset,
        "nodata": NODATA_VALUE,
    }
