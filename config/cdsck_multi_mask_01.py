"""单区域、多云掩膜的 C-DSCK 配置。

pipeline 会依次执行 ``cloud_masks`` 中的每项掩膜，并将结果文件名附加各项的
``name``。
"""

CONFIG = dict(
    mode="degraded",
    data=dict(
        # 同一研究区域的固定输入。
        coarse_path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/coarse",
        fine_path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/fine",
        label_path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/label",
        dates=["230705"],
        coarse_pattern="C{identifier}.tif",
        fine_pattern="F{identifier}.tif",
        label_pattern="L{identifier}.tif",

        cloud_masks=[
            dict(
                name="cloud_09",
                path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/mask_09",
                pattern="M{identifier}.tif",
            ),
            # dict(
            #     name="cloud_30",
            #     path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/mask_30",
            #     pattern="M{identifier}.tif",
            # ),
            # dict(
            #     name="cloud_10",
            #     path="data/c_dsck/landsat_sentinel2/lan_sen_degraded/mask_10",
            #     pattern="M{identifier}.tif",
            # ),
        ],
    ),
    methods=["c_dsck"],
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
    cdsck=dict(
        coarse_scale=3,
        fine_scale=2,
        coarse_window=1,
        fine_window=3,
        psf_sigma=1.0,
        cross_mode="degrade",
        max_points=100,
        max_radius=50,
        batch_size=512,
    ),
    backend=dict(
        mode="gpu",
    ),
    output=dict(
        directory="tmp/results/cdsck_multi_mask/lan_sen_degraded",
    ),
)
