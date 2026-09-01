# Kriging Boundary Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Complete variogram, support-kernel, and system-builder boundaries while preserving public ATPRK and DSCK output within the locked regression tolerance.

**Architecture:** `spatial.py` becomes the single source for spatial/variogram primitives and `support.py` becomes the sole owner of regularization/deconvolution kernels. `atprk.py` and `dsck.py` orchestrate those services; `systems.py` remains the fixed-system construction boundary.

**Tech Stack:** Python 3.10+, NumPy, SciPy least-squares, Numba, unittest.

**Spec:** `docs/superpowers/specs/2026-09-01-kriging-boundary-cleanup-design.md`

## Global Constraints

- Preserve public `ATPRKInterpolator` and `DSCKInterpolator` behavior.
- Preserve the locked public-output fixture with `rtol=1e-8` and `atol=1e-10`.
- Keep Numba kernels callable from Numba; do not replace `np.linalg.solve` with an explicit inverse.
- Keep old direct function names only as implementation-free compatibility aliases.

---

### Task 1: Separate self and cross variogram fitting

**Files:**
- Modify: `tests/test_variogram.py`
- Modify: `kriging/variogram.py`
- Modify: `kriging/dsck.py`

**Interfaces:**
- Produces `CrossVariogramEstimator.fit()` with a two-value fit and `fit_cross()` with a three-value fit.
- Produces `VariogramEstimator.fit()` defaulting to `model.residual` while retaining an explicit compatibility override.

- [x] **Step 1: Write the failing tests**

```python
def test_cross_estimator_retains_two_parameter_self_fit(self):
    fit = CrossVariogramEstimator().fit(
        np.arange(16.0).reshape(4, 4), 2,
        np.array([1.0, 2.0]), np.array([1.0, 1.0]),
    )
    self.assertEqual(fit.parameters.shape, (2,))

def test_estimator_uses_model_residual_without_override(self):
    model = RecordingSelfModel()
    VariogramEstimator(model=model).fit(np.zeros((3, 3)), 1,
                                        np.array([1.0]), np.array([1.0, 1.0]))
    self.assertTrue(model.was_called)
```

- [x] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_variogram -v`

Expected: inherited cross self-fit receives a cross residual and the custom model residual is not called.

- [x] **Step 3: Implement the minimal split**

```python
super().__init__(model=self_model or ExponentialVariogramModel(),
                 empirical_kernel=empirical_kernel,
                 residual_kernel=self_residual_kernel)
self.cross_model = cross_model or ExponentialCrossVariogramModel()
self._cross_residual_kernel = cross_residual_kernel or self.cross_model.residual
```

Make `fit_cross()` use `_cross_residual_kernel`; preserve optional explicit residual overrides for DSCK compatibility.

- [x] **Step 4: Verify GREEN and public outputs**

Run: `.venv/bin/python -m unittest tests.test_variogram tests.test_regression_baseline -v`

- [x] **Step 5: Commit**

```bash
git add kriging/variogram.py kriging/dsck.py tests/test_variogram.py
git commit -m "fix: separate self and cross variogram fitting"
```

### Task 2: Move support-scale kernels below algorithms

**Files:**
- Modify: `tests/test_support.py`
- Modify: `kriging/support.py`
- Modify: `kriging/atprk.py`
- Modify: `kriging/dsck.py`
- Modify: `kriging/spatial.py`

**Interfaces:**
- Produces `support.atprk_regularization`, `support.atprk_deconvolution`, `support.regularization_{coarse,cross,fine}`, and `support.deconvolution_{coarse,cross,fine}`.
- Consumes Numba-compatible `spatial.exponential_variogram` and `spatial.exponential_cross_variogram`.

- [x] **Step 1: Write failing ownership/equivalence tests**

```python
def test_support_owns_dsck_deconvolution_kernels(self):
    self.assertEqual(support.deconvolution_fine.__module__, "kriging.support")
    actual = support.deconvolution_fine(1, 2, np.array([2.0, 4.0]),
                                        0.5, 0.5, 1, 1, 0.1)
    np.testing.assert_allclose(actual, np.array([1.2, 2.4]), rtol=0, atol=0)
```

- [x] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_support -v`

Expected: the kernel is still defined by `kriging.dsck`.

- [x] **Step 3: Implement lower dependency direction**

Move ATPRK's area regularization/deconvolution and DSCK's six regularization/deconvolution kernels to `support.py`. Import canonical spatial kernels there. Replace algorithm-local bodies with imports/aliases. Replace DSCK's duplicate spatial bodies with imports from `spatial.py`; retain legacy DSCK operation ordering in `spatial.py` before deleting the copies.

- [x] **Step 4: Verify focused and public-output tests**

Run: `.venv/bin/python -m unittest tests.test_support tests.test_synthetic_smoke tests.test_regression_baseline -v`

- [x] **Step 5: Commit**

```bash
git add kriging/support.py kriging/spatial.py kriging/atprk.py kriging/dsck.py tests/test_support.py
git commit -m "refactor: move support kernels below algorithms"
```

### Task 3: Route legacy matrix helper through the builder

**Files:**
- Modify: `tests/test_systems.py`
- Modify: `kriging/systems.py`
- Modify: `kriging/dsck.py`

**Interfaces:**
- Produces `DSCKSystemBuilder.build(coarse_variogram, cross_variogram, fine_variogram)`.
- Produces `dsck._build_kriging_system(coarse_variogram, cross_variogram, fine_variogram)` as the legacy helper's tested adapter to the shared builder.
- Produces `dsck.calculate_matrix()` whose matrix and each RHS use the shared builder.

- [x] **Step 1: Write a failing adapter test**

```python
def test_legacy_dsck_adapter_returns_shared_system(self):
    system = dsck._build_kriging_system(np.eye(1), np.ones((1, 1)), np.eye(1))
    np.testing.assert_array_equal(system.matrix[-2:],
                                  np.array([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0]]))
```

- [x] **Step 2: Verify RED**

Run: `.venv/bin/python -m unittest tests.test_systems -v`

Expected: `dsck._build_kriging_system` does not exist before implementation.

- [x] **Step 3: Implement builder routing**

Rename builder arguments to `*_variogram`. Add `_build_kriging_system()` that returns `DSCKSystemBuilder.build(...)`. In `calculate_matrix()`, call that adapter rather than manually using `hstack`/`vstack`, and create constraints with `DSCKSystemBuilder.rhs(...)`. Keep its existing `(matrix, vectors)` return type for `tools/cal_kringing_matrix.py`.

- [x] **Step 4: Verify full suite**

Run: `.venv/bin/python -m unittest discover -v`

- [x] **Step 5: Commit**

```bash
git add kriging/systems.py kriging/dsck.py tests/test_systems.py
git commit -m "refactor: route legacy DSCK matrix helper through builder"
```

### Task 4: Document the stable CPU boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-01-kriging-boundary-cleanup-design.md`
- Modify: `docs/superpowers/plans/2026-09-01-kriging-boundary-cleanup.md`

- [x] **Step 1: Add a focused architecture note**

State that GPU/backend work is intentionally deferred and the public interpolator interface stays stable while `spatial.py`, `support.py`, and `systems.py` can acquire backend-specific implementations later.

- [x] **Step 2: Verify repository state**

Run: `git diff --check && .venv/bin/python -m unittest discover -v && git status --short`

Expected: no whitespace errors and no test failures.

- [x] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-09-01-kriging-boundary-cleanup-design.md docs/superpowers/plans/2026-09-01-kriging-boundary-cleanup.md
git commit -m "docs: describe stable kriging numerical boundaries"
```
