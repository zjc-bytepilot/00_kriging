# Kriging Boundary Cleanup Design

## Goal

Finish the incomplete CPU-side boundary extraction without changing the public
interpolator API or materially changing ATPRK/DSCK results. The locked
regression fixture remains the acceptance gate (`rtol=1e-8`, `atol=1e-10`).

## Ownership

`spatial.py` is the single owner of stateless spatial and variogram primitives:
padding, PSF generation, downsampling, empirical self/cross semivariograms,
and exponential model/residual kernels. Its variogram implementations retain
the legacy DSCK operation order so the one-lag least-squares path stays stable.

`support.py` owns support-scale regularization and deconvolution kernels for
both methods. It may depend on `spatial.py`; `atprk.py` and `dsck.py` import
from it and must not be imported by it. Algorithm modules keep orchestration
and kriging coefficient/RHS kernels only.

## Variogram fitting

`VariogramEstimator.fit()` uses its model's `residual` by default, so selecting
a model changes fitting behavior. An optional residual-kernel override remains
for numerical compatibility. `CrossVariogramEstimator` initialises its parent
with a two-parameter self model/residual, separately stores a three-parameter
cross model/residual, and uses the latter only in `fit_cross()`.
Its original four positional constructor arguments and public `.model` meaning
(the cross model) remain compatible; new self/cross extension points are
keyword-only.

## Kriging systems

System-builder arguments describe support variograms, not covariances. The
legacy `calculate_matrix()` remains public for tooling compatibility but builds
its coefficient matrix and RHS values through `DSCKSystemBuilder`.

## Compatibility and verification

Keep legacy import names as aliases only where direct callers could use them;
do not duplicate implementation bodies. Add behavior tests for the cross
estimator's inherited self-fit, model-default residual dispatch, support-kernel
ownership/equivalence, and the legacy matrix adapter. Run the complete suite
and locked public-output regression tests after every boundary migration.
