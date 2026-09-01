# Kriging 架构与性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持现有 CLI 与插值器调用方式不变的前提下，分离变异函数、支撑尺度和克里金系统职责，并减少 ATPRK/DSCK 的重复矩阵工作。

**Architecture:** 公开插值器继续位于 `estimators.py`，仅负责输入校验、PSF 缓存和算法编排。`variogram.py` 提供自/交叉变异函数拟合，`support.py` 保存可 JIT 编译的正则化/反卷积核，`systems.py` 构建固定克里金矩阵并使用 `numpy.linalg.solve`。`atprk.py` 与 `dsck.py` 保留兼容入口，但委托给这些组件。

**Tech Stack:** Python 3.10, NumPy, SciPy `least_squares`, Numba, Rasterio, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-01-kriging-architecture-performance-design.md`

## Global Constraints

- 公开的 `ATPRKInterpolator`、`DSCKInterpolator`、`Sharpening.py` 和 `config/*.py` 调用方式不得变化。
- 最终 ATPRK/DSCK 回归比较必须满足 `rtol=1e-8, atol=1e-10`。
- 保持 `float64` 主计算；不得改变 PSF、变异函数公式、反卷积搜索空间或 `least_squares` 初值/参数。
- 继续使用模块级 `@jit(nopython=True)` 数值核；不要把 Numba 热循环改成实例方法。
- 本阶段不实现 GPU/CUDA。
- 所有测试命令使用隔离环境：`.venv/bin/python -m unittest`。

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `requirements.txt` | 可复现的核心运行/测试依赖版本下界。 |
| `kriging/variogram.py` | 指数自/交叉模型、实验变异函数采样和拟合结果对象。 |
| `kriging/support.py` | JIT 支撑尺度正则化与反卷积核，不含公开编排。 |
| `kriging/systems.py` | ATPRK/DSCK 固定矩阵、RHS 构造和求解器。 |
| `kriging/spatial.py` | 共享 PSF/降采样原语与旧变异函数名兼容导出。 |
| `kriging/atprk.py` | ATPRK 的回归残差流程和旧 `ATPRK_Sharpen` 兼容入口。 |
| `kriging/dsck.py` | DSCK 的拟合/预测流程和旧 `DSCK_Regression_Sharpen` 兼容入口。 |
| `kriging/estimators.py` | 公开对象 API，缓存与进度回调。 |
| `tests/test_*.py` | 变异函数、系统、兼容 API 和端到端回归测试。 |
| `tests/fixtures/kriging_regression_baseline.npz` | 修改代码前生成的确定性 ATPRK/DSCK 金标准输出。 |
| `tools/benchmark_kriging.py` | 小尺寸预热后基准，记录但不硬编码机器相关阈值。 |

## Task 1: Lock the reproducible baseline

**Files:**
- Create: `requirements.txt`
- Create: `tests/test_regression_baseline.py`
- Create: `tests/fixtures/kriging_regression_baseline.npz`
- Modify: `tests/test_synthetic_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `synthetic_pair`, `SearchConfig`, `ATPRKInterpolator`, `DSCKInterpolator`.
- Produces: `load_regression_baseline() -> dict[str, np.ndarray]` and stable fixture keys `atprk_prediction`, `atprk_uncertainty`, `dsck_prediction`.

- [ ] **Step 1: Write the missing-fixture regression test**

```python
def load_regression_baseline() -> dict[str, np.ndarray]:
    fixture = Path(__file__).with_name("fixtures") / "kriging_regression_baseline.npz"
    with np.load(fixture) as values:
        return {name: values[name] for name in values.files}

def test_current_public_algorithms_match_golden_baseline(self) -> None:
    coarse, fine = synthetic_pair()
    search = SearchConfig(0.5, 0.5, 0.5, 1, 1, 1, 0.1, 1)
    atprk = ATPRKInterpolator(ATPRKConfig(window=1, psf_sigma=1.0), search)
    dsck = DSCKInterpolator(DSCKConfig(3, 2, 1, 1, 1.0), search)
    actual_atprk = atprk.sharpen(coarse, fine, band_count=1)
    actual_dsck = dsck.sharpen(coarse, fine, band_count=1)
    baseline = load_regression_baseline()
    np.testing.assert_allclose(actual_atprk.prediction, baseline["atprk_prediction"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(actual_dsck, baseline["dsck_prediction"], rtol=1e-8, atol=1e-10)
```

- [ ] **Step 2: Run the regression test and verify it fails because the fixture does not exist**

Run: `.venv/bin/python -m unittest tests.test_regression_baseline -v`

Expected: `FileNotFoundError` for `tests/fixtures/kriging_regression_baseline.npz`.

- [ ] **Step 3: Generate the fixture from the unmodified public API**

Create a local generation block in the test module that uses `synthetic_pair(size=15, scale=3)`, the existing one-step `SearchConfig`, `ATPRKConfig(window=1, psf_sigma=1.0)`, and `DSCKConfig(coarse_scale=3, fine_scale=2, coarse_window=1, fine_window=1, psf_sigma=1.0)`. Save only these arrays:

```python
np.savez_compressed(
    fixture_path,
    atprk_prediction=atprk_result.prediction,
    atprk_uncertainty=atprk_result.uncertainty,
    dsck_prediction=dsck_prediction,
)
```

Run the generation block once before changing any numerical code; remove the generation block after the fixture has been committed so tests are read-only.

- [ ] **Step 4: Run all existing and regression tests**

Run: `.venv/bin/python -m unittest discover -v`

Expected: all smoke tests and the new golden-output test pass.

- [ ] **Step 5: Add reproducible dependency instructions**

Create `requirements.txt`:

```text
numpy>=2.2,<2.6
scipy>=1.15,<1.16
numba>=0.67,<0.68
rasterio==1.4.4
```

Add to the README:

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

- [ ] **Step 6: Commit the baseline**

```bash
git add requirements.txt README.md tests/test_synthetic_smoke.py tests/test_regression_baseline.py tests/fixtures/kriging_regression_baseline.npz
git commit -m "test: lock kriging numerical baseline"
```

## Task 2: Extract typed variogram fitting services

**Files:**
- Create: `kriging/variogram.py`
- Modify: `kriging/spatial.py`
- Test: `tests/test_variogram.py`

**Interfaces:**
- Consumes: `spatial.semivariogram`, `spatial.cross_semivariogram`, `scipy.optimize.least_squares`.
- Produces: `VariogramFit`, `ExponentialVariogramModel`, `ExponentialCrossVariogramModel`, `VariogramEstimator`, `CrossVariogramEstimator`.

- [ ] **Step 1: Write failing model and estimator tests**

```python
def test_exponential_models_match_legacy_spatial_functions() -> None:
    h = np.array([0.0, 1.0, 3.0])
    np.testing.assert_array_equal(
        ExponentialVariogramModel().evaluate(np.array([2.0, 4.0]), h),
        spatial.exponential_variogram(np.array([2.0, 4.0]), h),
    )

def test_cross_estimator_uses_expected_lags_and_empirical_values() -> None:
    first = np.arange(16.0).reshape(4, 4)
    second = first * 2.0
    fit = CrossVariogramEstimator().fit_cross(
        first, second, max_lag=2, distances=np.array([3, 6]), initial=np.array([10.0, 100.0, 1.0])
    )
    np.testing.assert_array_equal(fit.lags, np.array([3, 6]))
    assert fit.parameters.shape == (3,)
```

- [ ] **Step 2: Run the new tests and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_variogram -v`

Expected: `ModuleNotFoundError: No module named 'kriging.variogram'`.

- [ ] **Step 3: Implement the variogram module without changing formulas**

```python
@dataclass(frozen=True)
class VariogramFit:
    parameters: np.ndarray
    lags: np.ndarray
    empirical_values: np.ndarray

class VariogramEstimator:
    def empirical(self, plane: np.ndarray, max_lag: int) -> np.ndarray:
        return np.asarray([spatial.semivariogram(plane, lag) for lag in range(1, max_lag + 1)])

    def fit(self, plane: np.ndarray, max_lag: int, distances: np.ndarray, initial: np.ndarray) -> VariogramFit:
        empirical = self.empirical(plane, max_lag)
        result = least_squares(spatial.exponential_variogram_residual, initial, args=(distances, empirical))
        return VariogramFit(np.asarray(result.x), np.asarray(distances), empirical)

class CrossVariogramEstimator(VariogramEstimator):
    def empirical_cross(self, first: np.ndarray, second: np.ndarray, max_lag: int) -> np.ndarray:
        return np.asarray([spatial.cross_semivariogram(first, second, lag) for lag in range(1, max_lag + 1)])

    def fit_cross(self, first: np.ndarray, second: np.ndarray, max_lag: int, distances: np.ndarray, initial: np.ndarray) -> VariogramFit:
        empirical = self.empirical_cross(first, second, max_lag)
        result = least_squares(spatial.exponential_cross_variogram_residual, initial, args=(distances, empirical))
        return VariogramFit(np.asarray(result.x), np.asarray(distances), empirical)
```

`evaluate` and residual methods must delegate to the existing exponential formulas in `spatial.py`; retain `myfun`, `myfun_fit`, `myfun2`, and `myfun2_fit` there as compatibility aliases.

- [ ] **Step 4: Run unit and golden regression tests**

Run: `.venv/bin/python -m unittest tests.test_variogram tests.test_regression_baseline -v`

Expected: all tests pass with no fixture changes.

- [ ] **Step 5: Commit the variogram service**

```bash
git add kriging/variogram.py kriging/spatial.py tests/test_variogram.py
git commit -m "feat: add typed variogram estimators"
```

## Task 3: Separate support-scale JIT kernels

**Files:**
- Create: `kriging/support.py`
- Modify: `kriging/atprk.py`
- Modify: `kriging/dsck.py`
- Test: `tests/test_support.py`

**Interfaces:**
- Consumes: exponential model functions from `kriging.spatial` and the existing Numba loop order.
- Produces: `atprk_deconvolution`, `dsck_deconvolution_coarse`, `dsck_deconvolution_fine`, `dsck_deconvolution_cross` plus their JIT regularization helpers.

- [ ] **Step 1: Write output-equivalence tests for each moved kernel**

```python
def test_support_reexports_the_stable_psf_and_downsampling_primitives() -> None:
    assert support.gaussian_psf is spatial.gaussian_psf
    assert support.downsample_plane is spatial.downsample_plane

def test_atprk_deconvolution_matches_pre_extraction_reference() -> None:
    actual = support.atprk_deconvolution(1, 3, np.array([2.0, 4.0]), 0.5, 0.5, 1, 1, 0.1)
    np.testing.assert_allclose(actual, np.array([1.2, 2.4]), rtol=0, atol=0)

def test_dsck_cross_deconvolution_returns_three_parameters() -> None:
    actual = support.dsck_deconvolution_cross(1, 3, 2, np.array([1.0, 2.0, 4.0]), 0.5, 0.5, 0.5, 1, 1, 1, 0.1)
    np.testing.assert_allclose(actual, np.array([0.6, 1.2, 2.4]), rtol=0, atol=0)
```

- [ ] **Step 2: Run the tests and verify the support module is absent**

Run: `.venv/bin/python -m unittest tests.test_support -v`

Expected: `ModuleNotFoundError: No module named 'kriging.support'`.

- [ ] **Step 3: Move only support-scale pure functions to `support.py`**

Move the following functions verbatim, retaining decorators, argument order, dtypes and loop nesting:

```text
atprk.py: r_area_area2, ATP_deconvolution
dsck.py: regularization_coarse, deconvolution_coarse,
         regularization_fine, deconvolution_fine,
         regularization_cross, deconvolution_cross
```

Expose the shared PSF, extension and downsampling primitives through the new boundary without making a second implementation:

```python
from .spatial import downsample_plane, extend_plane, gaussian_psf

__all__ = [
    "atprk_deconvolution", "downsample_plane", "dsck_deconvolution_coarse",
    "dsck_deconvolution_cross", "dsck_deconvolution_fine", "extend_plane", "gaussian_psf",
]
```

Use explicit imports in the legacy modules:

```python
from .support import atprk_deconvolution
from .support import dsck_deconvolution_coarse, dsck_deconvolution_cross, dsck_deconvolution_fine
```

Keep aliases with the old function names in `atprk.py` and `dsck.py` so direct scripts continue working.

- [ ] **Step 4: Verify each kernel and both golden algorithm outputs**

Run: `.venv/bin/python -m unittest tests.test_support tests.test_regression_baseline -v`

Expected: bit-identical support-kernel checks and `rtol=1e-8, atol=1e-10` algorithm checks pass.

- [ ] **Step 5: Commit support extraction**

```bash
git add kriging/support.py kriging/atprk.py kriging/dsck.py tests/test_support.py
git commit -m "refactor: centralize support scale kernels"
```

## Task 4: Build and solve fixed kriging systems once

**Files:**
- Create: `kriging/systems.py`
- Modify: `kriging/atprk.py`
- Modify: `kriging/dsck.py`
- Test: `tests/test_systems.py`

**Interfaces:**
- Consumes: covariance blocks from existing ATPRK/DSCK JIT covariance functions.
- Produces: `KrigingSolver.solve(matrix, rhs)`, `ATPRKSystemBuilder.build(covariance)`, `DSCKSystemBuilder.build(coarse_covariance, cross_covariance, fine_covariance)`.

- [ ] **Step 1: Write failing linear-system and matrix-layout tests**

```python
def test_solver_matches_numpy_solve() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([[1.0], [2.0]])
    np.testing.assert_allclose(KrigingSolver.solve(matrix, rhs), np.linalg.solve(matrix, rhs), rtol=0, atol=0)

def test_atprk_builder_adds_one_unbiasedness_constraint() -> None:
    system = ATPRKSystemBuilder().build(np.eye(2))
    np.testing.assert_array_equal(system.matrix[-1], np.array([1.0, 1.0, 0.0]))

def test_dsck_builder_adds_two_constraints() -> None:
    system = DSCKSystemBuilder().build(np.eye(2), np.zeros((2, 1)), np.eye(1))
    assert system.matrix.shape == (5, 5)
```

- [ ] **Step 2: Run the tests and verify the systems module is absent**

Run: `.venv/bin/python -m unittest tests.test_systems -v`

Expected: `ModuleNotFoundError: No module named 'kriging.systems'`.

- [ ] **Step 3: Implement fixed matrix data objects and builders**

```python
@dataclass(frozen=True)
class KrigingSystem:
    matrix: np.ndarray

class KrigingSolver:
    @staticmethod
    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(matrix, rhs)
```

Build ATPRK as `[[rVV, ones], [ones.T, 0]]`. Build DSCK as the four covariance blocks plus the existing two unbiasedness rows. Add `atprk_rhs(rv_v)` and `dsck_rhs(rv_v, ru_v)` helpers that append the same constraint values currently used by the legacy code.

- [ ] **Step 4: Integrate builders before per-subpixel loops**

In `atprk.calculate_parameter`, compute `TVV` and `system = ATPRKSystemBuilder().build(TVV)` once before `for i in range(s)`. Inside the loop build only `rvV` and call `KrigingSolver.solve(system.matrix, ATPRKSystemBuilder.rhs(rvV))`.

In `dsck.calculate_parameter`, compute `r1`, `r2`, `r4` and `system = DSCKSystemBuilder().build(r1, r2, r4)` once before `for i in range(s0)`. Inside the loop build only `r5`, `r6` and call `KrigingSolver.solve(system.matrix, DSCKSystemBuilder.rhs(r5, r6))`.

Remove `np.linalg.inv` from both modules. If Numba cannot compile calls through Python classes, keep the outer `calculate_parameter` as regular Python and preserve all covariance/RHS Numba functions unchanged.

- [ ] **Step 5: Run system, smoke and golden tests**

Run: `.venv/bin/python -m unittest tests.test_systems tests.test_synthetic_smoke tests.test_regression_baseline -v`

Expected: system tests pass; all numerical outputs remain within the global tolerance.

- [ ] **Step 6: Commit system construction optimization**

```bash
git add kriging/systems.py kriging/atprk.py kriging/dsck.py tests/test_systems.py
git commit -m "perf: reuse fixed kriging systems"
```

## Task 5: Use shared estimators and retain compatibility wrappers

**Files:**
- Modify: `kriging/atprk.py`
- Modify: `kriging/dsck.py`
- Modify: `kriging/estimators.py`
- Modify: `kriging/__init__.py`
- Test: `tests/test_algorithm_integration.py`

**Interfaces:**
- Consumes: `VariogramEstimator`, `CrossVariogramEstimator`, support kernels and system builders.
- Produces: unchanged `ATPRK_Sharpen`, `DSCK_Regression_Sharpen`, `ATPRKInterpolator.sharpen`, and `DSCKInterpolator.sharpen` signatures.

- [ ] **Step 1: Write failing integration tests for delegation and PSF reuse**

```python
def test_atprk_reuses_cached_psf_for_equal_scale() -> None:
    model = ATPRKInterpolator(config, search)
    first = model._psf_for_scale(3)
    second = model._psf_for_scale(3)
    assert first is second

def test_public_legacy_functions_and_interpolators_match_golden_baseline() -> None:
    coarse, fine = synthetic_pair()
    baseline = load_regression_baseline()
    atprk_prediction = ATPRK_Sharpen(
        coarse[:, :, 0], fine[:, :, 0], 0.5, 0.5, 1, 1, 0.1, 1, 1, gaussian_psf(3, 1, 1.0)
    )[1]
    dsck_prediction = DSCK_Regression_Sharpen(
        coarse[:, :, 0], fine[:, :, 0], 0.5, 0.5, 0.5, 1, 1, 1, 0.1, 1, 1, 1,
        gaussian_psf(3, 1, 1.0), gaussian_psf(2, 1, 1.0), 3, 2,
    )
    np.testing.assert_allclose(atprk_prediction, baseline["atprk_prediction"][:, :, 0], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(dsck_prediction, baseline["dsck_prediction"][:, :, 0], rtol=1e-8, atol=1e-10)
```

- [ ] **Step 2: Run the integration tests and verify the cache method is absent**

Run: `.venv/bin/python -m unittest tests.test_algorithm_integration.AlgorithmIntegrationTest.test_atprk_reuses_cached_psf_for_equal_scale -v`

Expected: `AttributeError: 'ATPRKInterpolator' object has no attribute '_psf_for_scale'`.

- [ ] **Step 3: Replace local fitting code with shared services**

In `ATPRK_Sharpen`, replace the direct list comprehension and direct `least_squares` call with:

```python
fit = VariogramEstimator().fit(
    RB,
    max_lag=H,
    distances=np.arange(s, s * H + 1, s),
    initial=np.array([100.0, 1.0]),
)
```

In DSCK, use one `VariogramEstimator` for coarse/fine fits and one `CrossVariogramEstimator` for the cross fit. Preserve the existing distances and initial vectors exactly. Keep `ATPRK_Sharpen` and `DSCK_Regression_Sharpen` as functions with their current signatures that call the new pipeline.

- [ ] **Step 4: Add safe PSF caching to public interpolators**

```python
class ATPRKInterpolator:
    def __init__(self, config: ATPRKConfig, search: SearchConfig) -> None:
        self.config = config
        self.search = search
        self._psf_by_scale: dict[int, np.ndarray] = {}

    def _psf_for_scale(self, scale: int) -> np.ndarray:
        return self._psf_by_scale.setdefault(
            scale, gaussian_psf(scale, self.config.window, self.config.psf_sigma)
        )
```

Use `_psf_for_scale` in `sharpen_band`. Preserve DSCK's eager two-PSF cache and do not cache variance-dependent fitted matrices across unrelated bands.

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m unittest discover -v`

Expected: all tests pass, including golden output checks and public API validation.

- [ ] **Step 6: Commit integration refactor**

```bash
git add kriging/atprk.py kriging/dsck.py kriging/estimators.py kriging/__init__.py tests/test_algorithm_integration.py
git commit -m "refactor: compose shared kriging services"
```

## Task 6: Add a repeatable performance benchmark and finish documentation

**Files:**
- Create: `tools/benchmark_kriging.py`
- Modify: `README.md`
- Test: `tests/test_benchmark_cli.py`

**Interfaces:**
- Consumes: public interpolators and deterministic synthetic data.
- Produces: `python -m tools.benchmark_kriging --method {atprk,dsck,both} --repeats N` JSON timing output.

- [ ] **Step 1: Write the CLI output test**

```python
def test_benchmark_reports_selected_method_and_positive_elapsed_time() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "tools.benchmark_kriging", "--method", "atprk", "--repeats", "1"],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["atprk"]["seconds_per_run"] > 0
```

- [ ] **Step 2: Run the test and verify the module is absent**

Run: `.venv/bin/python -m unittest tests.test_benchmark_cli -v`

Expected: subprocess failure because `tools.benchmark_kriging` does not exist.

- [ ] **Step 3: Implement a deterministic, pre-warmed benchmark**

```python
def benchmark(method: str, repeats: int) -> dict[str, dict[str, float]]:
    coarse, fine = synthetic_pair(size=15, scale=3)
    runner = build_runner(method)
    runner()  # compile Numba and warm caches; excluded from timing
    durations = [time_call(runner) for _ in range(repeats)]
    return {method: {"seconds_per_run": float(np.mean(durations)), "runs": repeats}}
```

Use `json.dump(result, sys.stdout)` as the only stdout payload. Do not assert an absolute performance target in tests; the benchmark exists to compare before/after runs on the same host.

- [ ] **Step 4: Document commands and design boundaries**

Add README sections for: `.venv` creation from `requirements.txt`, baseline tests, benchmark invocation, preserved APIs, `solve` replacing inverse, and the explicit non-goal of GPU acceleration in this iteration.

- [ ] **Step 5: Run final verification**

Run: `.venv/bin/python -m unittest discover -v`

Run: `.venv/bin/python -m tools.benchmark_kriging --method both --repeats 2`

Expected: all tests pass; benchmark emits valid JSON with positive timings for both methods.

- [ ] **Step 6: Commit documentation and benchmark**

```bash
git add tools/benchmark_kriging.py tests/test_benchmark_cli.py README.md
git commit -m "docs: add kriging benchmark workflow"
```
