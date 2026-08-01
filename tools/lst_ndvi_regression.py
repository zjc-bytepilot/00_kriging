from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


DATA_ROOT = Path(__file__).resolve().parents[1] / '20251216'
COARSE_LST_PATH = DATA_ROOT / 'LST' / 'lst_10km' / 'lst_201401_10km.tif'
FINE_NDVI_PATH = (
    DATA_ROOT
    / 'ndvi'
    / 'NDVI_CLIP_1km'
    / 'idw_MYD13A3.A2014001.1_km_monthly_NDVI.tif'
)
REFERENCE_LST_PATH = DATA_ROOT / 'LST' / 'monthly' / 'lst_201401.tif'
OUTPUT_DIR = DATA_ROOT / 'regression_output'


def read_raster(path):
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype(np.float64)
        profile = dataset.profile.copy()
        transform = dataset.transform
        crs = dataset.crs
        bounds = dataset.bounds
    return array, profile, transform, crs, bounds


def aggregate_to_coarse_grid(fine_array, fine_transform, fine_crs,
                             coarse_shape, coarse_transform, coarse_crs):
    """将精细栅格按面积平均聚合到粗分辨率网格。"""
    coarse_array = np.full(coarse_shape, np.nan, dtype=np.float64)
    reproject(
        source=fine_array,
        destination=coarse_array,
        src_transform=fine_transform,
        src_crs=fine_crs,
        dst_transform=coarse_transform,
        dst_crs=coarse_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return coarse_array


def fit_linear_regression(ndvi, lst):
    valid = np.isfinite(ndvi) & np.isfinite(lst)
    x = ndvi[valid]
    y = lst[valid]
    if x.size < 2:
        raise ValueError('共同有效像元不足，无法进行线性回归。')

    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = np.sum(residual ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_total
    rmse = np.sqrt(np.mean(residual ** 2))
    return slope, intercept, r_squared, rmse, valid


def save_prediction(path, prediction, reference_profile):
    profile = reference_profile.copy()
    profile.update(dtype='float32', count=1, nodata=np.nan, compress='deflate')
    with rasterio.open(path, 'w', **profile) as dataset:
        dataset.write(prediction.astype(np.float32), 1)


def plot_results(coarse_lst, coarse_ndvi, fine_ndvi, predicted_lst,
                 reference_lst, regression_mask, slope, intercept,
                 r_squared, rmse, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    panels = [
        (axes[0, 0], coarse_lst, 'Observed LST (10 km)', 'coolwarm', 'LST'),
        (axes[0, 1], coarse_ndvi, 'Aggregated NDVI (10 km)', 'YlGn', 'NDVI'),
        (axes[1, 0], fine_ndvi, 'Original NDVI (1 km)', 'YlGn', 'NDVI'),
        (axes[1, 1], predicted_lst, 'Regression-predicted LST (1 km)', 'coolwarm', 'LST'),
        (axes[1, 2], reference_lst, 'Reference LST (1 km)', 'coolwarm', 'LST'),
    ]
    finite_lst = reference_lst[np.isfinite(reference_lst)]
    lst_limits = np.nanpercentile(finite_lst, [2, 98])
    for axis, image, title, cmap, label in panels:
        kwargs = {'cmap': cmap}
        if label == 'LST':
            kwargs.update(vmin=lst_limits[0], vmax=lst_limits[1])
        layer = axis.imshow(image, **kwargs)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(layer, ax=axis, shrink=0.78, label=label)

    scatter_axis = axes[0, 2]
    x = coarse_ndvi[regression_mask]
    y = coarse_lst[regression_mask]
    if x.size > 5000:
        indices = np.linspace(0, x.size - 1, 5000, dtype=int)
        x_plot, y_plot = x[indices], y[indices]
    else:
        x_plot, y_plot = x, y
    scatter_axis.scatter(x_plot, y_plot, s=8, alpha=0.25, edgecolors='none')
    regression_x = np.linspace(np.min(x), np.max(x), 200)
    scatter_axis.plot(regression_x, slope * regression_x + intercept, color='red', linewidth=2)
    scatter_axis.set_title(
        f'LST = {slope:.4f} × NDVI {intercept:+.4f}\n'
        f'$R^2$ = {r_squared:.4f}, RMSE = {rmse:.4f}'
    )
    scatter_axis.set_xlabel('Aggregated NDVI (10 km)')
    scatter_axis.set_ylabel('Observed LST (10 km)')
    scatter_axis.grid(alpha=0.2)

    figure.suptitle('January 2014 LST–NDVI Linear Regression', fontsize=16)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coarse_lst, _, coarse_transform, coarse_crs, _ = read_raster(COARSE_LST_PATH)
    fine_ndvi, fine_profile, fine_transform, fine_crs, _ = read_raster(FINE_NDVI_PATH)
    reference_lst, reference_profile, _, _, _ = read_raster(REFERENCE_LST_PATH)

    if fine_ndvi.shape != reference_lst.shape:
        raise ValueError('1 km NDVI 与 1 km LST 的栅格尺寸不一致。')

    coarse_ndvi = aggregate_to_coarse_grid(
        fine_ndvi,
        fine_transform,
        fine_crs,
        coarse_lst.shape,
        coarse_transform,
        coarse_crs,
    )
    slope, intercept, r_squared, coarse_rmse, regression_mask = fit_linear_regression(
        coarse_ndvi, coarse_lst
    )
    predicted_lst = slope * fine_ndvi + intercept
    predicted_lst[~np.isfinite(fine_ndvi)] = np.nan

    comparison_mask = np.isfinite(predicted_lst) & np.isfinite(reference_lst)
    reference_rmse = np.sqrt(
        np.mean((predicted_lst[comparison_mask] - reference_lst[comparison_mask]) ** 2)
    )
    reference_correlation = np.corrcoef(
        predicted_lst[comparison_mask], reference_lst[comparison_mask]
    )[0, 1]

    prediction_path = OUTPUT_DIR / 'predicted_lst_1km_201401.tif'
    figure_path = OUTPUT_DIR / 'lst_ndvi_regression_201401.png'
    report_path = OUTPUT_DIR / 'lst_ndvi_regression_201401.txt'
    save_prediction(prediction_path, predicted_lst, fine_profile)
    plot_results(
        coarse_lst,
        coarse_ndvi,
        fine_ndvi,
        predicted_lst,
        reference_lst,
        regression_mask,
        slope,
        intercept,
        r_squared,
        coarse_rmse,
        figure_path,
    )

    report = (
        f'slope={slope:.10f}\n'
        f'intercept={intercept:.10f}\n'
        f'coarse_r_squared={r_squared:.10f}\n'
        f'coarse_rmse={coarse_rmse:.10f}\n'
        f'coarse_valid_pixels={int(regression_mask.sum())}\n'
        f'reference_rmse={reference_rmse:.10f}\n'
        f'reference_correlation={reference_correlation:.10f}\n'
        f'reference_valid_pixels={int(comparison_mask.sum())}\n'
    )
    report_path.write_text(report, encoding='utf-8')
    print(report, end='')
    print(f'prediction={prediction_path}')
    print(f'figure={figure_path}')
    print(f'report={report_path}')


if __name__ == '__main__':
    main()
