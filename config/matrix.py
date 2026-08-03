"""训练影像的 DSCK 半方差矩阵配置。"""

from copy import deepcopy

from config.degraded import CONFIG as DEGRADED_CONFIG

CONFIG = deepcopy(DEGRADED_CONFIG)
CONFIG["data"].update(dict(
    coarse_path="data/train",
    fine_path="data/train",
    dates=["20230211", "20230705", "20230813", "20230928"],
))
CONFIG["methods"] = ["dsck"]
CONFIG["dsck"]["fine_window"] = 4
