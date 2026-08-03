# Kriging Downscaling

该项目提供两种克里金空间降尺度方法：

- `DSCKInterpolator`：双支撑协同克里金（DSCK）
- `ATPRKInterpolator`：面积到点回归克里金（ATPRK）

## 目录结构

```text
config/degraded.py       退化数据实验配置（包含标签评估）
config/real.py           真实数据运行配置（无需标签）
config/matrix.py         训练矩阵生成配置
kriging/config.py        类型化配置及校验
kriging/data.py          GeoTIFF 数据读取与结果保存
kriging/metrics.py       光谱质量指标
kriging/estimators.py    面向对象算法接口
kriging/pipeline.py      实验编排
kriging/dsck.py          DSCK 底层数值核
kriging/atprk.py         ATPRK 底层数值核
kriging/spatial.py       两种算法共用的 PSF、降采样和半变异函数
Sharpening.py            命令行入口
```

## 使用

配置文件使用普通 Python `CONFIG` 字典。退化数据实验会读取标签并计算指标：

```powershell
.venv\Scripts\python.exe Sharpening.py --config config\degraded.py
```

真实数据不要求标签，结果会直接保存到配置的输出目录：

```powershell
.venv\Scripts\python.exe Sharpening.py --config config\real.py
```

`file_pattern` 使用 `{identifier}` 代入 `dates` 中的值。例如
`C{identifier}.tif`、`F{identifier}.tif` 和 `L{identifier}.tif` 分别表示
coarse、fine 和 label。例如日期 `20231113` 会读取 `C20231113.tif`、
`F20231113.tif` 和 `L20231113.tif`。
配置中的路径均相对于项目根目录，例如 `data/test` 实际表示
`00_kriging/data/test`。因此运行命令时不依赖终端当前所在目录。

也可以直接在 Python 中使用算法类：

```python
from kriging import DSCKInterpolator, load_config

config = load_config("config/degraded.py")
model = DSCKInterpolator(config.dsck, config.search)
prediction = model.sharpen(coarse_cube, fine_cube, config.band_count)
```

训练影像的克里金矩阵通过独立配置生成：

```powershell
.venv\Scripts\python.exe -m tools.cal_kringing_matrix --config config\matrix.py
```

默认输出到被 Git 忽略的 `tmp/kriging_matrices/`。

## 栅格可视化

SMP 土壤湿度数据可使用内置预设绘图：

```powershell
.venv\Scripts\python.exe -m tools.visualize_raster `
  usa_smt-1\sm_20180101_new.tif --preset smp
```

默认图片保存到 `tmp/visualizations/`。其他单波段 GeoTIFF 可以使用
`--preset generic`，并通过 `--band`、`--cmap`、`--vmin`、`--vmax`、
`--title` 和 `--colorbar-label` 调整显示。
