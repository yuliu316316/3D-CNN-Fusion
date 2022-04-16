# -*- coding: utf-8 -*-
import time
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import losses
from utils import str2bool, count_params

import cnn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import scipy.io as io
arch_names = list(cnn.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('L1loss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='model',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cnn',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=True, type=str2bool)
    parser.add_argument('--dataset', default="jiu0Monkey",
                        help='dataset name')
    parser.add_argument('--input-channels', default=6, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')

    args = parser.parse_args()

    return args

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def normalize(image, mask=None):
    if mask is None:
        mask = image != image[0, 0, 0]

    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - image[mask].mean()) / image[mask].std()
    image[image == image.min()] = -9
    return image

def normalize_2047(image):
    image = image.astype(dtype=np.float32)
    image = image/2047
    image[image > 1.0] = 1.0
    return image

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    # create model
    print("=> creating model %s" %args.arch)
    model = cnn.__dict__[args.arch](args)

    model = nn.DataParallel(model)
    model = model.cuda()

    # get test MRI volumes
    # t1 = glob.glob('./source_images/*t1.mat')
    # t2 = glob.glob('./source_images/*t2.mat')
    t1ce = glob.glob('./source_images/*t1ce.mat')
    flair = glob.glob('./source_images/*flair.mat')

    slice = 100
    img1_path = t1ce[0]
    print(img1_path)
    #input_T1 = read_img(img1_path).astype(np.float32)
    input_T1 = io.loadmat(img1_path)
    input_T1 = input_T1['t1ce']

    #input_11 = np.lib.pad(input_T1, ((0, 0), (padding, padding), (padding, padding)), 'edge')
    #io.savemat('./result/T1_1.mat', {'T1': input_11})

    img2_path = flair[0]
    #input_T2 = read_img(img2_path).astype(np.float32)
    input_T2 = io.loadmat(img2_path)
    input_T2 = input_T2['flair']
    print(img2_path)

    plt.figure('T1')
    plt.imshow(input_T1[slice, :, :], cmap='gray')
    plt.figure('T2')
    plt.imshow(input_T2[slice, :, :], cmap='gray')

    assert np.array_equal(input_T1.shape, input_T2.shape), 'shapes do not match'
    assert input_T1[0, 0, 0] == 0, 'non-zero background?!'
    assert input_T2[0, 0, 0] == 0, 'non-zero background?!'

    mask = (input_T1 != 0) | (input_T2 != 0)

    # %% Extract the Brain Box
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))

    input_T1 = normalize_2047(input_T1)
    input_T2 = normalize_2047(input_T2)


    input_T11 = input_T1[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    input_T22 = input_T2[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    print(input_T11.shape)
    zz, xx, yy = input_T11.shape

    train_data_T1 = torch.tensor(input_T11, dtype=torch.float32)
    train_data_T2 = torch.tensor(input_T22, dtype=torch.float32)
    train_data_T1 = torch.reshape(train_data_T1, (1, 1, zz, xx, yy))
    train_data_T2 = torch.reshape(train_data_T2, (1, 1, zz, xx, yy))
    train_data_T1 = train_data_T1.cuda()
    train_data_T2 = train_data_T2.cuda()

    # model._initialize_weights()
    model.load_state_dict(torch.load('./%s/model.pth' % args.name))
    print(count_params(model))

    # compute output
    output = model(train_data_T1, train_data_T2)
    ones = np.ones([1, 1, zz, xx, yy])
    ones = torch.tensor(ones)
    ones = ones.float()
    ones = ones.cuda()
    output = output.mul(train_data_T1) + (ones - output).mul(train_data_T2)

    # outputs = output
    output = output.cpu()
    output = output.detach().numpy()
    print(output.shape)
    output = output.squeeze()
    output = np.lib.pad(output, ((minZidx, 155 - maxZidx), (minXidx, 240 - maxXidx), (minYidx, 240 - maxYidx)), 'edge')
    print(output.shape)
    output = output*2047

    plt.figure('result')
    plt.imshow(output[slice, :, :], cmap='gray')
    #io.savemat('./result/result_%s_t1ceflair.mat' %img1_path[-14:-9], {'result': output})
    plt.show()

if __name__ == '__main__':
    main()
