import random
import numpy as np
import h5py
import torch
import os
from os.path import join as opj
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def read_data_BraTs(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data1 = np.asarray(hf.get('data1'), dtype='float32')
        data2 = np.asarray(hf.get('data2'), dtype='float32')
        return data1, data2

def tf_fspecial_gauss_3d_torch(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=1)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=1)

    z_data = np.expand_dims(z_data, axis=0)
    z_data = np.expand_dims(z_data, axis=1)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(-((x**2 + y**2 + z**2)/(3.0*sigma**2)))
    return g / torch.sum(g)

def SSIM_3d_torch(img1, img2, k1=0.01, k2=0.03, L=2, window_size=11):
    """
    The function is to calculate the ssim score
    """
    window = tf_fspecial_gauss_3d_torch(window_size, 1.5)
    window = window.cuda()
    mu1 = torch.nn.functional.conv3d(img1, window, stride=1, padding=0)
    mu2 = torch.nn.functional.conv3d(img2, window, stride=1, padding=0)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.nn.functional.conv3d(img1 * img1, window, stride=1, padding=0) - mu1_sq
    sigma2_sq = torch.nn.functional.conv3d(img2 * img2, window, stride=1, padding=0) - mu2_sq
    sigma1_2 = torch.nn.functional.conv3d(img1 * img2, window, stride=1, padding=0) - mu1_mu2
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1-torch.mean(ssim_map)

def random_crop(datadir, batch_size, image_size):
    # 随机读取数据 *********************************
    batchsize = batch_size
    size = image_size
    dir = datadir
    list_dirs = []
    p = os.listdir(dir)
    for subject in p:
        list_dirs.append((opj(dir, subject), subject))

    t2_input = []
    flair_input = []
    gt_input = []
    for i in range(batchsize):
        dir = random.choice(list_dirs)
        dir_ = dir[0]
        a = np.load(dir_)
        t2 = a[2]
        flair = a[3]
        gt = a[4]
        xx, yy, zz = t2.shape
        x = np.random.randint(0, xx - size)
        y = np.random.randint(0, yy - size)
        z = np.random.randint(0, zz - size)
        t2_ = t2[x:x + size, y:y + size, z:z + size]
        flair_ = flair[x:x + size, y:y + size, z:z + size]
        gt_ = gt[x:x + size, y:y + size, z:z + size]

        # plt.figure()
        # plt.imshow(t2[:, :, 60], cmap='gray')
        # plt.figure()
        # plt.imshow(flair[:, :, 60], cmap='gray')
        # plt.show()

        t2_1 = t2_[np.newaxis, :, :, :]
        flair_1 = flair_[np.newaxis, :, :, :]
        gt_1 = gt_[np.newaxis, :, :, :]
        if i == 0:
            t2_input = t2_1
            flair_input = flair_1
            gt_input = gt_1
        else:
            t2_input = np.append(t2_input, t2_1, axis=0)
            flair_input = np.append(flair_input, flair_1, axis=0)
            gt_input = np.append(gt_input, gt_1, axis=0)

    return t2_input, flair_input, gt_input