# -*- coding: utf-8 -*-
import time
import os
import argparse
from glob import glob
import numpy as np
import joblib
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from utils import str2bool, count_params
import pandas as pd
import cnn
from utils import (
  read_data_BraTs,
  SSIM_3d_torch
)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='3D-CNN-Fusion',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cnn')
    parser.add_argument('--deepsupervision', default=True, type=str2bool)

    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)

    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=10, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=3e-2, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args

def train(args, input1, input2, model, optimizer, epoch, log, scheduler=None):
    model.train()
    target1 = input1
    target2 = input2
    target1 = torch.tensor(target1)
    target2 = torch.tensor(target2)
    target1 = target1.cuda()
    target2 = target2.cuda()

    # compute output of the mask
    output = model(target1, target2)

    # compute output
    ones = np.ones([1, 1, 80, 80, 80])
    ones = torch.tensor(ones)
    ones = ones.float().cuda()
    # ones = ones.cuda()
    outputs = output.mul(target1) + (ones - output).mul(target2)
    # outputs = output

    loss1 = 2 * SSIM_3d_torch(target1, outputs) + SSIM_3d_torch(target2, outputs)     # ssim loss
    loss2 = torch.norm(outputs - target1) + 2 * torch.norm(outputs - target2)         # pixel loss
    loss1 = 450*loss1
    loss = loss1 + loss2

    # compute gradient and do optimizing step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_total = loss.cpu().detach().numpy()
    loss_ssim = loss1.cpu().detach().numpy()
    loss_norm = loss2.cpu().detach().numpy()

    return loss_ssim, loss_norm, loss_total

def main():
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % args.arch)
    model = cnn.__dict__[args.arch](args)
    #model = nn.DataParallel(model)
    model = model.cuda()

    print(count_params(model))
    print('# model parameters:', sum(param.numel() for param in model.parameters()))

    # get training data
    T1T2_t1, T1T2_t2 = read_data_BraTs('./Train_data/train_T1t2_size80.h5')
    T1T2_t1 = T1T2_t1.reshape((6030, 1, 80, 80, 80))
    T1T2_t2 = T1T2_t2.reshape((6030, 1, 80, 80, 80))

    T1ceFlair_t1ce, T1ceFlair_flair = read_data_BraTs('./Train_data/train_T1ceflair_size80.h5')
    T1ceFlair_t1ce = T1ceFlair_t1ce.reshape((6030, 1, 80, 80, 80))
    T1ceFlair_flair = T1ceFlair_flair.reshape((6030, 1, 80, 80, 80))

    #model._initialize_weights()
    #model.load_state_dict(torch.load('models/%s/model.pth' %args.name))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    log = pd.DataFrame(index=[], columns=['epoch', 'loss_SSIM', 'loss_norm', 'loss_total'])

    ssim_loss1 = []
    norm_loss1 = []
    total_loss1 = []

    ssim_loss3 = []
    norm_loss3 = []
    total_loss3 = []

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        batch_idxs = len(T1T2_t1) // args.batch_size
        print(batch_idxs)
        count = 0
        for idx in range(0, batch_idxs):
            count = count + 1
            # train for one epoch
            batch_images_T1 = T1T2_t1[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_images_T2 = T1T2_t2[idx * args.batch_size: (idx + 1) * args.batch_size]
            ssimloss1, normloss1, totalloss1 = train(args, batch_images_T1, batch_images_T2, model, optimizer, epoch, log)

            batch_images_T1 = T1ceFlair_t1ce[idx * args.batch_size: (idx + 1) * args.batch_size]
            batch_images_T2 = T1ceFlair_flair[idx * args.batch_size: (idx + 1) * args.batch_size]
            ssimloss3, normloss3, totalloss3 = train(args, batch_images_T1, batch_images_T2, model, optimizer, epoch, log)

            if count % 10==0:
                print('epoch:', epoch, 'step:', count, 'loss_total:', totalloss1, 'loss_ssim:', ssimloss1, 'loss_norm:', normloss1)
                ssim_loss1.append(ssimloss1)
                norm_loss1.append(normloss1)
                total_loss1.append(totalloss1)

                ssim_loss3.append(ssimloss3)
                norm_loss3.append(normloss3)
                total_loss3.append(totalloss3)


        # save trained model
        torch.save(model.state_dict(), 'models/%s/%s_model.pth' %(args.name, epoch))
        torch.cuda.empty_cache()

    LOSS = {'ssimloss1': ssim_loss1, 'normloss1': norm_loss1, 'totalloss1': total_loss1,  \
            'ssimloss3': ssim_loss3, 'normloss3': norm_loss3, 'totalloss3': total_loss3}
    df1 = pd.DataFrame(LOSS)
    df1.to_csv("./models/%s/loss.csv" %(args.name), index=False)

if __name__ == '__main__':
    main()
