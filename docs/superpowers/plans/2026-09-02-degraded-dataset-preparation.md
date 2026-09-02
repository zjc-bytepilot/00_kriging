# 批量退化数据集生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立、可批量执行的 GeoTIFF 退化数据集生成脚本，按文件名序号配对 GF6 与 Landsat 数据。

**Architecture:** 在 `tools/prepare_degraded_dataset.py` 中定义无 PyTorch 依赖的 `DegradedPairDataset`、`DegradationProcessor` 和 CLI。Dataset 负责基于第一个下划线前的数字配对并验证输入；Processor 对每个波段直接调用既有 `downsample_plane`，不重投影、不对齐网格，并将结果写为 C/F/L GeoTIFF 及 manifest。

**Tech Stack:** Python 3、NumPy、Rasterio、现有 `kriging.spatial`、unittest。

**Spec:** `docs/superpowers/designs/2026-09-02-degraded-dataset-preparation.md`

## Global Constraints

- 只新增预处理模块、脚本和测试，不修改 ATPRK、DSCK、pipeline 或配置加载逻辑。
- 配对键是文件名第一个下划线前的连续数字；日期不参与配对。
- coarse 和 fine 的每个波段必须调用 `kriging.spatial.downsample_plane`，默认 `scale=3`、`window=1`、`psf_sigma=1.0`。
- 禁止重投影、裁剪和网格对齐；下采样文件保留源原点，像元尺寸增大 `scale` 倍。
- 默认拒绝覆盖输出；仅 `--overwrite` 可覆盖。

---

### Task 1: 数据集发现与序号配对

**Files:**

- Create: `tools/prepare_degraded_dataset.py`
- Create: `tests/test_prepare_degraded_dataset.py`

**Interfaces:**

- Produces: `DegradedPair(serial: str, gf6_path: Path, landsat_path: Path)` dataclass.
- Produces: `DegradedPairDataset(input_root: str | Path)` with `pairs: tuple[DegradedPair, ...]`, `__len__() -> int`, `__iter__() -> Iterator[DegradedPair]`.

- [ ] **Step 1: Write the failing pairing tests**

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_prepare_degraded_dataset.py -v`

Expected: FAIL because `tools.prepare_degraded_dataset` does not exist.

- [ ] **Step 3: Implement minimal discovery and validation**

```python
SERIAL_PATTERN = re.compile(r"^(?P<serial>\d+)_")

@dataclass(frozen=True)
class DegradedPair:
    serial: str
    gf6_path: Path
    landsat_path: Path

class DegradedPairDataset:
    def __init__(self, input_root: str | Path) -> None:
        root = Path(input_root)
        gf6 = _index_by_serial(root / "gf6")
        landsat = _index_by_serial(root / "landsat")
        if gf6.keys() != landsat.keys():
            raise ValueError("GF6 与 Landsat 存在缺少配对的序号。")
        self.pairs = tuple(
            DegradedPair(serial, gf6[serial], landsat[serial])
            for serial in sorted(gf6, key=int)
        )
```

- [ ] **Step 4: Run the pairing tests to verify they pass**

Run: `pytest tests/test_prepare_degraded_dataset.py -v`

Expected: PASS for the pairing and missing-pair tests.

- [ ] **Step 5: Commit**

```bash
git add tools/prepare_degraded_dataset.py tests/test_prepare_degraded_dataset.py
git commit -m "feat: add serial-paired degradation dataset"
```

### Task 2: 逐波段退化与 GeoTIFF 输出

**Files:**

- Modify: `tools/prepare_degraded_dataset.py`
- Modify: `tests/test_prepare_degraded_dataset.py`

**Interfaces:**

- Consumes: `DegradedPair` from Task 1 and `kriging.spatial.downsample_plane`.
- Produces: `DegradationProcessor(scale: int = 3, window: int = 1, psf_sigma: float = 1.0)`.
- Produces: `process_pair(pair: DegradedPair, output_root: str | Path, overwrite: bool = False) -> dict[str, str]`.

- [ ] **Step 1: Write the failing processing test**

```python
def test_processor_writes_expected_triplet_without_grid_alignment(tmp_path: Path) -> None:
    gf6 = np.arange(36, dtype=np.float32).reshape(6, 6, 1)
    landsat = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    pair = write_pair(tmp_path, "01", gf6, landsat)

    output = DegradationProcessor(scale=2, window=1, psf_sigma=1.0).process_pair(
        pair, tmp_path / "out"
    )

    psf = gaussian_psf(scale=2, window=1, sigma=1.0)
    assert_allclose(read_hwc(output["fine"]), downsample_cube(gf6, 2, 1, psf))
    assert_allclose(read_hwc(output["coarse"]), downsample_cube(landsat, 2, 1, psf))
    assert_allclose(read_hwc(output["label"]), landsat)
```

- [ ] **Step 2: Run the processing test to verify it fails**

Run: `pytest tests/test_prepare_degraded_dataset.py::test_processor_writes_expected_triplet_without_grid_alignment -v`

Expected: FAIL because `DegradationProcessor` does not exist.

- [ ] **Step 3: Implement minimal per-band processing and profile scaling**

```python
def _downsample_cube(values: np.ndarray, *, scale: int, window: int, psf: np.ndarray) -> np.ndarray:
    return np.stack(
        [downsample_plane(values[:, :, band], scale, window, psf) for band in range(values.shape[2])],
        axis=-1,
    )

def _scaled_profile(profile: Mapping[str, Any], values: np.ndarray, scale: int) -> dict[str, Any]:
    result = dict(profile)
    result.update(height=values.shape[0], width=values.shape[1], transform=profile["transform"] * Affine.scale(scale))
    return result
```

The writer must save `fine/F{serial}.tif`, `coarse/C{serial}.tif`, and
`label/L{serial}.tif`. Before writing, it must reject any existing target
unless `overwrite=True`.

- [ ] **Step 4: Run the processing test to verify it passes**

Run: `pytest tests/test_prepare_degraded_dataset.py::test_processor_writes_expected_triplet_without_grid_alignment -v`

Expected: PASS, including exact `downsample_plane` values and source-origin-preserving transforms.

- [ ] **Step 5: Commit**

```bash
git add tools/prepare_degraded_dataset.py tests/test_prepare_degraded_dataset.py
git commit -m "feat: generate degraded GeoTIFF triplets"
```

### Task 3: 批量 CLI、manifest 与回归验证

**Files:**

- Modify: `tools/prepare_degraded_dataset.py`
- Modify: `tests/test_prepare_degraded_dataset.py`

**Interfaces:**

- Consumes: `DegradedPairDataset` and `DegradationProcessor`.
- Produces: `main(argv: Sequence[str] | None = None) -> int` and `<output-root>/manifest.json`.

- [ ] **Step 1: Write the failing CLI test**

```python
def test_cli_processes_all_pairs_and_records_source_dates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_pair_with_dates(tmp_path / "input", "01", "20230418", "20190418")
    write_pair_with_dates(tmp_path / "input", "02", "20230520", "20190521")

    assert main(["--input-root", str(tmp_path / "input"), "--output-root", str(tmp_path / "out")]) == 0

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert [entry["serial"] for entry in manifest["pairs"]] == ["01", "02"]
    assert manifest["pairs"][0]["gf6_source"].endswith("01_20230418.tif")
    assert manifest["pairs"][0]["landsat_source"].endswith("01_20190418.tif")
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `pytest tests/test_prepare_degraded_dataset.py::test_cli_processes_all_pairs_and_records_source_dates -v`

Expected: FAIL because `main` does not create `manifest.json`.

- [ ] **Step 3: Implement the argparse entrypoint and manifest writer**

```python
parser.add_argument("--input-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--scale", type=int, default=3)
parser.add_argument("--window", type=int, default=1)
parser.add_argument("--psf-sigma", type=float, default=1.0)
parser.add_argument("--overwrite", action="store_true")
```

The JSON manifest must contain `scale`, `window`, `psf_sigma`, and a `pairs`
list ordered by numeric serial. Each pair entry must contain `serial`,
`gf6_source`, `landsat_source`, `fine`, `coarse`, and `label` paths.

- [ ] **Step 4: Run the focused test file and full suite**

Run: `pytest tests/test_prepare_degraded_dataset.py -v && pytest -q`

Expected: all new tests pass and all existing tests remain green.

- [ ] **Step 5: Commit**

```bash
git add tools/prepare_degraded_dataset.py tests/test_prepare_degraded_dataset.py
git commit -m "feat: add degradation dataset preparation CLI"
```

## Self-Review

- Spec coverage: Task 1 implements serial pairing and input validation; Task 2 implements the required direct `downsample_plane` processing, no-alignment profile behavior, output names, and overwrite protection; Task 3 implements multi-pair invocation and provenance manifest.
- Placeholder scan: all task steps name concrete files, APIs, commands, assertions, output names, and errors; no TBD/TODO markers remain.
- Type consistency: Task 1's `DegradedPair` feeds Task 2; Task 2's `process_pair` dictionary feeds Task 3's manifest construction; CLI arguments exactly match the design document.
