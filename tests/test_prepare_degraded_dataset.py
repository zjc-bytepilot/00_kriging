import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from numpy.testing import assert_allclose

from kriging.spatial import downsample_plane, gaussian_psf
from tools.prepare_degraded_dataset import (
    DegradationProcessor,
    DegradedPair,
    DegradedPairDataset,
    main,
)


def write_tiff(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(str(value).encode())


def write_geotiff(path: Path, values: np.ndarray, *, transform: Affine) -> Path:
    """Write a real HWC float GeoTIFF and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=values.shape[2],
        dtype=values.dtype,
        transform=transform,
        crs="EPSG:4326",
    ) as dataset:
        dataset.write(np.moveaxis(values, -1, 0))
    return path


def write_pair(
    root: Path,
    serial: str,
    gf6: np.ndarray,
    landsat: np.ndarray,
) -> DegradedPair:
    return DegradedPair(
        serial=serial,
        gf6_path=write_geotiff(
            root / "gf6" / f"{serial}_gf6.tif",
            gf6,
            transform=Affine.translation(100, 200) @ Affine.scale(3, -3),
        ),
        landsat_path=write_geotiff(
            root / "landsat" / f"{serial}_landsat.tif",
            landsat,
            transform=Affine.translation(-20, 40) @ Affine.scale(30, -30),
        ),
    )


def read_hwc(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return np.moveaxis(dataset.read(), 0, -1)


def downsample_cube(values: np.ndarray, scale: int, window: int, psf: np.ndarray) -> np.ndarray:
    return np.stack(
        [downsample_plane(values[:, :, band], scale, window, psf) for band in range(values.shape[2])],
        axis=-1,
    )


def test_dataset_pairs_files_by_serial_not_date(tmp_path: Path) -> None:
    write_tiff(tmp_path / "gf6" / "01_20230418.tif", value=1)
    write_tiff(tmp_path / "landsat" / "01_20190418.tif", value=2)

    dataset = DegradedPairDataset(tmp_path)

    assert [(pair.serial, pair.gf6_path.name, pair.landsat_path.name) for pair in dataset] == [
        ("01", "01_20230418.tif", "01_20190418.tif")
    ]


def test_dataset_rejects_missing_pair(tmp_path: Path) -> None:
    write_tiff(tmp_path / "gf6" / "01_20230418.tif", value=1)

    with pytest.raises(ValueError, match="缺少配对"):
        DegradedPairDataset(tmp_path)


def test_dataset_rejects_duplicate_serial(tmp_path: Path) -> None:
    write_tiff(tmp_path / "gf6" / "01_20230418.tif", value=1)
    write_tiff(tmp_path / "gf6" / "01_20230518.tif", value=1)
    write_tiff(tmp_path / "landsat" / "01_20190418.tif", value=2)

    with pytest.raises(ValueError, match="重复"):
        DegradedPairDataset(tmp_path)


def test_dataset_iterates_pairs_in_numeric_serial_order(tmp_path: Path) -> None:
    write_tiff(tmp_path / "gf6" / "10_20230418.tif", value=1)
    write_tiff(tmp_path / "gf6" / "2_20230418.tif", value=1)
    write_tiff(tmp_path / "landsat" / "10_20190418.tif", value=2)
    write_tiff(tmp_path / "landsat" / "2_20190418.tif", value=2)

    assert [pair.serial for pair in DegradedPairDataset(tmp_path)] == ["2", "10"]


def test_dataset_rejects_malformed_tiff_filename(tmp_path: Path) -> None:
    write_tiff(tmp_path / "gf6" / "not-a-serial.tif", value=1)
    write_tiff(tmp_path / "landsat" / "01_20190418.tif", value=2)

    with pytest.raises(ValueError, match="命名规则"):
        DegradedPairDataset(tmp_path)


def test_dataset_ignores_tiff_directories(tmp_path: Path) -> None:
    (tmp_path / "gf6" / "01_placeholder.tif").mkdir(parents=True)
    write_tiff(tmp_path / "landsat" / "01_20190418.tif", value=2)

    with pytest.raises(ValueError, match="缺少配对"):
        DegradedPairDataset(tmp_path)


def test_processor_writes_expected_triplet_without_grid_alignment(tmp_path: Path) -> None:
    gf6 = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
    landsat = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    pair = write_pair(tmp_path, "01", gf6, landsat)

    output = DegradationProcessor(scale=2, window=1, psf_sigma=1.0).process_pair(
        pair, tmp_path / "out"
    )

    psf = gaussian_psf(scale=2, window=1, sigma=1.0)
    assert output == {
        "fine": str(tmp_path / "out" / "fine" / "F01.tif"),
        "coarse": str(tmp_path / "out" / "coarse" / "C01.tif"),
        "label": str(tmp_path / "out" / "label" / "L01.tif"),
    }
    assert_allclose(read_hwc(output["fine"]), downsample_cube(gf6, 2, 1, psf))
    assert_allclose(read_hwc(output["coarse"]), downsample_cube(landsat, 2, 1, psf))
    assert_allclose(read_hwc(output["label"]), landsat)
    with rasterio.open(output["fine"]) as fine:
        assert fine.transform == Affine.translation(100, 200) @ Affine.scale(6, -6)
    with rasterio.open(output["coarse"]) as coarse:
        assert coarse.transform == Affine.translation(-20, 40) @ Affine.scale(60, -60)


def test_processor_rejects_existing_outputs_without_overwrite(tmp_path: Path) -> None:
    gf6 = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
    landsat = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    pair = write_pair(tmp_path, "02", gf6, landsat)
    processor = DegradationProcessor(scale=2, window=1, psf_sigma=1.0)
    processor.process_pair(pair, tmp_path / "out")

    with pytest.raises(FileExistsError, match="已存在"):
        processor.process_pair(pair, tmp_path / "out")

    overwritten = processor.process_pair(pair, tmp_path / "out", overwrite=True)
    assert Path(overwritten["fine"]).is_file()


@pytest.mark.parametrize(
    ("gf6_shape", "landsat_shape", "match"),
    [
        ((7, 8, 1), (4, 4, 1), "整除"),
        ((8, 8, 1), (5, 4, 1), "整除"),
        ((8, 8, 2), (4, 4, 1), "波段"),
        ((8, 8, 1), (2, 4, 1), "形状"),
    ],
)
def test_processor_rejects_incompatible_shapes_before_writing(
    tmp_path: Path,
    gf6_shape: tuple[int, int, int],
    landsat_shape: tuple[int, int, int],
    match: str,
) -> None:
    pair = write_pair(
        tmp_path,
        "03",
        np.zeros(gf6_shape, dtype=np.float32),
        np.zeros(landsat_shape, dtype=np.float32),
    )
    output_root = tmp_path / "out"

    with pytest.raises(ValueError, match=match):
        DegradationProcessor(scale=2).process_pair(pair, output_root)

    assert not output_root.exists()


def test_cli_processes_all_pairs_and_records_source_dates(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    for serial, gf6_date, landsat_date in (
        ("01", "20230418", "20190418"),
        ("02", "20230520", "20190521"),
    ):
        write_geotiff(
            input_root / "gf6" / f"{serial}_{gf6_date}.tif",
            np.zeros((8, 8, 1), dtype=np.float32),
            transform=Affine.translation(100, 200) @ Affine.scale(3, -3),
        )
        write_geotiff(
            input_root / "landsat" / f"{serial}_{landsat_date}.tif",
            np.zeros((4, 4, 1), dtype=np.float32),
            transform=Affine.translation(-20, 40) @ Affine.scale(30, -30),
        )

    output_root = tmp_path / "out"
    assert main(["--input-root", str(input_root), "--output-root", str(output_root), "--scale", "2"]) == 0

    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["scale"] == 2
    assert [entry["serial"] for entry in manifest["pairs"]] == ["01", "02"]
    assert manifest["pairs"][0]["gf6_source"].endswith("01_20230418.tif")
    assert manifest["pairs"][0]["landsat_source"].endswith("01_20190418.tif")
    assert Path(manifest["pairs"][1]["fine"]).is_file()
