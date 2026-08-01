import torch
import DSCK as GSFD
import numpy as np
import Common_functions as fun


if __name__ == "__main__":
    # 加载train数据
    Coarse_path = r'D:\南卿\深度学习\Data\train\L'
    Fine_path = r'D:\南卿\深度学习\Data\train\S'
    Label_path = r'D:\南卿\深度学习\Data\label\L'
    data_name = [20230211, 20230705, 20230813, 20230928]

    s = 2
    s0 = 3
    W1 = 1
    W2 = 4
    sigma = s / 2
    # PSF图像退化和卷积
    PSF1 = GSFD.PSF(s0, W1, sigma)  # Coarse_imagePSF,12阶方阵
    PSF2 = GSFD.PSF(s, W2, sigma)
    PSF11 = GSFD.PSF(s0, W1, sigma)  # Fine_imagePSF,10阶方阵
    # 克里金插值参数
    Constant_min = 0.5
    Sill_min = 0.5
    Range_min = 0.5
    L_sill = 30
    L_range = 30
    L_constant = 30
    rate = 0.1
    H = 30

    Matrix_left = []
    Matrix_right = []

    Coarse_images = fun.load_data_numpy(data_name, Coarse_path, 'Landsat_up')  # 粗糙分辨率数据
    Fine_images = fun.load_data_numpy(data_name, Fine_path, 'Sentinel_up')  # 精细分辨率数据
    Label_images = fun.load_data_numpy(data_name, Label_path, 'Landsat')  # 标签数据
    for k in range(4):
        Coarse_image = Coarse_images[k]
        Fine_image = Fine_images[k]
        for i in range(4):
            matrix_left, Vector = GSFD.calculate_matrix(Coarse_image[:, :, i], Fine_image[:, :, i], Constant_min, Sill_min,
                                                    Range_min, L_sill, L_range, L_constant, rate, H, W1, W2, PSF1, PSF2)
            arrays = [lst for lst in Vector]
            matrix_right = np.concatenate(arrays, axis=1)
            Matrix_left.append(torch.tensor(matrix_left, dtype=torch.float32).unsqueeze(0))
            Matrix_right.append(torch.tensor(matrix_right, dtype=torch.float32).unsqueeze(0))
            print(f'克里金半方差矩阵计算进度:{k * 4 + i + 1} / {16}')

    Matrix_batch_left = torch.cat(Matrix_left, dim=0)
    Matrix_batch_right = torch.cat(Matrix_right, dim=0)

    torch.save(Matrix_batch_left, 'Matrix_batch_left.pt')
    torch.save(Matrix_batch_right, 'Matrix_batch_right.pt')