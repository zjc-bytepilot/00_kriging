import time

import numpy as np

import ATPRK as GSFA
import Common_functions as fun
import DSCK as GSFD


# 数据配置
Coarse_path = r'D:\南卿\深度学习\Data\test\L'
Fine_path = r'D:\南卿\深度学习\Data\test\S'
Label_path = r'D:\南卿\深度学习\Data\label\L'
data_name = [20231113]
# data_name = [20240525]

# 空间尺度参数
s = 2
s0 = 3

# 克里金插值窗口与 PSF 参数
W1 = 1
W2 = 3
sigma = s / 2

# 克里金插值参数
Constant_min = 0.5
Sill_min = 0.5
Range_min = 0.5
L_sill = 30
L_range = 30
L_constant = 30
rate = 0.1
H = 30

BAND_COUNT = 4


def load_images():
    """读取实验所需的粗分辨率、精细分辨率和标签影像。"""
    coarse_images = fun.load_data_numpy(data_name, Coarse_path, 'Landsat_up')
    fine_images = fun.load_data_numpy(data_name, Fine_path, 'Sentinel_up')
    label_images = fun.load_data_numpy(data_name, Label_path, 'Landsat')
    return coarse_images, fine_images, label_images


def select_experiment_images(coarse_images, fine_images, label_images):
    """选择当前实验影像，并在计算前给出明确的数据错误。"""
    image_groups = {
        '粗分辨率影像': coarse_images,
        '精细分辨率影像': fine_images,
        '标签影像': label_images,
    }
    for name, images in image_groups.items():
        if not images:
            raise FileNotFoundError(f'未加载到{name}，请检查数据路径和日期。')
        if images[0].ndim != 3 or images[0].shape[2] < BAND_COUNT:
            raise ValueError(
                f'{name}形状应为 (高度, 宽度, 至少 {BAND_COUNT} 个波段)，'
                f'实际为 {images[0].shape}。'
            )
    return coarse_images[0], fine_images[0], label_images[0]


def build_psfs():
    """构造 DSCK 和 ATPRK 共用的点扩散函数。"""
    psf1 = GSFD.PSF(s0, W1, sigma)
    psf2 = GSFD.PSF(s, W2, sigma)
    return psf1, psf2


def run_dsck(coarse_image, fine_image, output_template, psf1, psf2):
    """逐波段执行 DSCK 锐化，返回锐化结果和运行时间。"""
    sharpened = np.zeros_like(output_template)
    start_time = time.time()

    for band in range(BAND_COUNT):
        sharpened[:, :, band] = GSFD.DSCK_Regression_Sharpen(
            coarse_image[:, :, band],
            fine_image[:, :, band],
            Constant_min,
            Sill_min,
            Range_min,
            L_sill,
            L_range,
            L_constant,
            rate,
            H,
            W1,
            W2,
            psf1,
            psf2,
        )
        print(f'DSCK Sharpening进度：{band + 1}/{BAND_COUNT}')

    return sharpened, time.time() - start_time


def run_atprk(coarse_image, fine_image, output_template, psf1):
    """逐波段执行 ATPRK 锐化，返回误差、锐化结果和运行时间。"""
    sharpened = np.zeros_like(output_template)
    uncertainty = np.zeros_like(output_template)
    start_time = time.time()

    for band in range(BAND_COUNT):
        uncertainty_band, sharpened_band = GSFA.ATPRK_Sharpen(
            coarse_image[:, :, band],
            fine_image[:, :, band],
            Sill_min,
            Range_min,
            L_sill,
            L_range,
            rate,
            H,
            W1,
            psf1,
        )
        uncertainty[:, :, band] = uncertainty_band
        sharpened[:, :, band] = sharpened_band
        print(f'ATPRK Sharpening进度：{band + 1}/{BAND_COUNT}')

    return uncertainty, sharpened, time.time() - start_time


def print_evaluation(method_name, label_image, sharpened, elapsed_time):
    """计算并输出一种锐化方法的质量指标与耗时。"""
    rmse, cc, ergas, uiqi, sam = fun.evaluate_relation_spectral(
        label_image, sharpened, s0
    )
    print(f'{method_name}精度评定')
    print(f'RMSE为：{rmse}')
    print(f'CC为：{cc}')
    print(f'ERGAS为：{ergas}')
    print(f'UIQI为：{uiqi}')
    print(f'SAM为：{sam}')
    print(f'{method_name} Execution Time: {elapsed_time:.2f} seconds')


def main():
    coarse_images, fine_images, label_images = load_images()
    coarse_image, fine_image, label_image = select_experiment_images(
        coarse_images, fine_images, label_images
    )
    psf1, psf2 = build_psfs()

    z_dsck, dsck_elapsed = run_dsck(
        coarse_image, fine_image, label_image, psf1, psf2
    )
    print_evaluation('DSCK', label_image, z_dsck, dsck_elapsed)

    _, z_atprk, atprk_elapsed = run_atprk(
        coarse_image, fine_image, label_image, psf1
    )
    print_evaluation('ATPRK', label_image, z_atprk, atprk_elapsed)


if __name__ == '__main__':
    main()
