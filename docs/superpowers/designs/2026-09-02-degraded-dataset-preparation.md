# 批量退化数据集生成设计

## 目标

为任意数量的 GF6 与 Landsat GeoTIFF 文件生成退化实验所需的
`coarse`、`fine`、`label` 三类数据。实现必须是独立预处理能力，不能
改变既有 ATPRK、DSCK、pipeline 或配置加载的行为。

## 输入组织与配对

原始目录固定为两个传感器目录：

```text
<input-root>/
├── gf6/
│   ├── 01_20230418.tif
│   └── 02_20230520.tif
└── landsat/
    ├── 01_20190418.tif
    └── 02_20190521.tif
```

文件名第一个下划线前的连续数字是配对序号。日期不参与配对，因此
`gf6/01_20230418.tif` 与 `landsat/01_20190418.tif` 是一组。
每个目录中每个序号必须恰好出现一次；缺失配对、重复序号和不符合命名
规则的 `.tif` 文件都是输入错误。

## 数据处理

`DegradedPairDataset` 是一个轻量 Python 集合，不依赖 PyTorch。它负责：

1. 发现两个目录的 GeoTIFF；
2. 按序号构建 `DegradedPair`；
3. 对缺失、重复、无效命名进行 fail-fast 校验。

`DegradationProcessor` 负责单组影像的处理：

1. 使用 `gaussian_psf(scale, window, sigma)` 创建 PSF；
2. 对每个 GF6 波段调用 `downsample_plane`，得到 fine；
3. 对每个 Landsat 波段调用同一函数，得到 coarse；
4. 将原 Landsat 像元值作为 label。

不重投影、不裁剪、不执行网格对齐。降采样影像仅沿各自源影像的仿射变换
放大像元尺寸 `scale` 倍，保留源影像原点。这样不会把 GF6 的网格改写为
Landsat 网格。

## 输出与可追溯性

```text
<output-root>/
├── coarse/C01.tif
├── fine/F01.tif
├── label/L01.tif
└── manifest.json
```

输出文件以序号作为 identifier，便于现有退化配置使用
`dates=["01", "02"]` 与 `C{identifier}.tif`、`F{identifier}.tif`、
`L{identifier}.tif`。`manifest.json` 保存每一组的源文件路径、输出路径和
`scale`、`window`、`psf_sigma`，保留两个传感器各自的日期信息。

默认拒绝覆盖既有输出；调用方必须传入 `--overwrite` 才允许覆盖。

## 命令行接口

```bash
python -m tools.prepare_degraded_dataset \
  --input-root data/lan_gf \
  --output-root data/lan_gf_degraded \
  --scale 3 --window 1 --psf-sigma 1.0
```

脚本使用标准库参数解析，按序号顺序批量执行并输出 JSON 摘要。

## 验证

测试采用临时 GeoTIFF：

- Dataset 能按序号而非日期配对；
- `process_pair` 的 coarse/fine 每一波段数值与直接调用
  `downsample_plane` 相同；
- label 保持原 Landsat 像元值；
- 输出尺寸、仿射变换和文件命名正确；
- 缺失配对与重复序号产生明确错误。

