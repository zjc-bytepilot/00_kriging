# Kriging 架构与性能优化设计

## 目标

在不改变 CLI、配置格式和公开插值器接口的前提下，重构 ATPRK 与 DSCK 的共享数学职责，并优化重复计算和线性方程求解。新实现相对于当前基线的数值误差必须满足 `numpy.testing.assert_allclose(rtol=1e-8, atol=1e-10)`。

本阶段不引入 GPU、改变 PSF 定义、改变变异函数公式、改变 `least_squares` 初值/参数，也不改变 `float64` 主计算路径。

## 当前问题

- `spatial.py` 已保存共享的变异函数和退化原语，但 DSCK 仍有一套重复实现。
- `atprk.py` 与 `dsck.py` 都混合了拟合、反卷积、矩阵组装、逐像素 RHS 构建和求解。
- 两个算法在每个子像素上重复构建相同系统矩阵，并使用显式求逆 `inv(A) @ b`。

## 目标结构

```text
estimators.py  public API and algorithm orchestration
      |
      +-- variogram.py  model evaluation, empirical estimates, fitting results
      +-- support.py    PSF/downsampling and Numba regularization/deconvolution kernels
      +-- systems.py    ATPRK/DSCK fixed-system assembly, RHS builders, linear solver
      +-- spatial.py    backwards-compatible aliases for stable numerical primitives

atprk.py / dsck.py  backwards-compatible wrappers around the new components
```

`ATPRKInterpolator` 和 `DSCKInterpolator` 是并列的应用服务，不建立继承关系。二者通过组合重用变异函数估计器、支撑尺度核和系统构建器；DSCK 单独组合交叉变异函数能力。

## 组件职责

### `variogram.py`

- `ExponentialVariogramModel`：二维参数自变异模型及残差函数。
- `ExponentialCrossVariogramModel`：三维参数交叉变异模型及残差函数。
- `VariogramEstimator`：计算实验自变异函数并用既有 `least_squares` 调用拟合。
- `CrossVariogramEstimator`：在 `VariogramEstimator` 的基础上增加实验交叉变异和拟合。
- 不可变结果对象保存参数、lag 距离和实验值，避免匿名 `ndarray` 在流程中失去语义。

### `support.py`

- 保留现有 PSF、边界扩展、下采样及反卷积的公式与 Numba 编译边界。
- 将 ATPRK 和 DSCK 使用的支撑尺度/反卷积数值核集中在这里；高性能循环仍为模块级纯函数，不改为实例方法。
- 预计算不依赖影像值的局部坐标和权重，供同一模型实例的多波段调用复用。

### `systems.py`

- `ATPRKSystemBuilder` 构建固定的 `rVV`、约束行和每个细像素的 RHS。
- `DSCKSystemBuilder` 构建 `rVV`、`rVU`、`rUV`、`rUU` 和双约束行，并分别构建各子像素 RHS。
- `KrigingSolver` 使用 `numpy.linalg.solve(A, b)`，不显式构造逆矩阵。
- 固定矩阵每个波段只构建一次；若矩阵与参数不变，缓存预计算结果。RHS 仍逐子像素生成，以保持算法语义。

## 数据流

### ATPRK

```text
coarse + fine
  -> regression / residual
  -> VariogramEstimator.fit(residual)
  -> support deconvolution
  -> ATPRKSystemBuilder.build(fixed matrix)
  -> RHS per fine subpixel + KrigingSolver.solve
  -> regression surface + interpolated residual
```

### DSCK

```text
coarse + fine
  -> coarse/fine self-variograms + cross-variogram
  -> three support deconvolutions
  -> DSCKSystemBuilder.build(fixed block matrix)
  -> RHS per fine subpixel + KrigingSolver.solve
  -> combined prediction
```

## 兼容性与错误处理

- `Sharpening.py`、`config/*.py`、`ATPRKInterpolator`、`DSCKInterpolator` 的调用方式不变。
- 保留 `atprk.py` / `dsck.py` 的现有函数名作为兼容入口；它们转调新组件而不是维持第二套公式。
- 输入维度、缩放比例和波段数校验继续由公开插值器处理；新构建器对奇异或形状不匹配的系统矩阵给出含方法名与矩阵形状的异常。

## 性能策略

1. 以 `solve` 替代 `inv(A) @ b`，减少计算量和数值不稳定性。
2. 将每个子像素循环外的矩阵组装移出循环。
3. 缓存 PSF、局部坐标、固定矩阵与必要的分解/权重。
4. 将 DSCK 对 `spatial.py` 变异函数的重复实现收敛为单一来源。
5. 保持 Numba 热循环为纯函数；只有经基准确认等价的操作才进行向量化。

## 验证

- 保留现有端到端合成烟雾测试。
- 新增变异函数自/交叉实验值、拟合结果和系统求解的单元测试。
- 新增旧兼容入口与新编排器的回归对比：ATPRK 与 DSCK 最终结果均采用 `rtol=1e-8, atol=1e-10`。
- 新增小尺寸基准，分别记录冷启动后和预热后的每波段耗时；性能断言只比较固定硬件上的相对改进，不把绝对秒数写死。

## 非目标

- GPU/CUDA 改造。
- PSF、变异函数或去卷积搜索空间的数学修改。
- 配置文件格式、GeoTIFF I/O 或指标计算重写。
- 为了面向对象而把 Numba 纯数值核改成对象成员函数。
