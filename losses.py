import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
    
class L1loss(nn.Module):
    def __init__(self):
        super(L1loss, self).__init__()

    def forward(self, input, target1, target2):
        l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
        l1_loss = l1_loss_fn(input, target1)+l1_loss_fn(input, target2)
#      
    
        return l1_loss
    
class SSIM_loss(nn.Module):
    def __init__(self):
        super(SSIM_loss, self).__init__()
        

    def forward(self,img1, img2):
        """
        The function is to calculate the ssim score
        """
        k1=0.01
        k2=0.03
        L=2
        window_size=11
        size=window_size
        sigma=1.5
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
        
        x=g / torch.sum(g)
        
        window = x
        window = window.cuda(1)
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
        
        ssimloss = 1-torch.mean(ssim_map) 
        
        return ssimloss  
        
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
    
    


