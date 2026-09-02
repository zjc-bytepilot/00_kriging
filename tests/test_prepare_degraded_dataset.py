from pathlib import Path

import pytest
from tools.prepare_degraded_dataset import DegradedPairDataset


def write_tiff(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(str(value).encode())


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
