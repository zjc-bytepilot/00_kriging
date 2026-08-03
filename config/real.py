"""真实数据配置：无需参考标签，直接输出降尺度结果。"""

CONFIG = dict(
    mode="real",
    data=dict(
        coarse_path="data/real",
        fine_path="data/real",
        dates=["20231113"],
        coarse_pattern="C{identifier}.tif",
        fine_pattern="F{identifier}.tif",
    ),
    methods=["dsck", "atprk"],
    band_count=4,
    search=dict(
        constant_min=0.5,
        sill_min=0.5,
        range_min=0.5,
        sill_steps=30,
        range_steps=30,
        constant_steps=30,
        step_size=0.1,
        max_lag=30,
    ),
    dsck=dict(
        coarse_scale=3,
        fine_scale=2,
        coarse_window=1,
        fine_window=3,
        psf_sigma=1.0,
    ),
    atprk=dict(
        window=1,
        psf_sigma=1.0,
    ),
    output=dict(
        directory="tmp/results/real",
    ),
)
