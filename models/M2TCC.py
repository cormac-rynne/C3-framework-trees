import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from config import cfg
from misc import layer
import models


class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name,loss_1_fn,loss_2_fn,sigma=None):
        super(CrowdCounter, self).__init__()        

        # if model_name == 'SANet':
        #     from .M2TCC_Model.SANet import SANet as net
        net = getattr(models, model_name)

        print('SANet:',cfg.LAMBDA_1)

        self.CCN = net()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
        self.loss_1_fn = loss_1_fn.cuda()
        self.loss_2_fn = loss_2_fn.cuda()

        gs_layer = getattr(layer, 'Gaussianlayer')
        if sigma is None:
            sigma = cfg.SIGMA
        kernel_size = 6 * sigma + 1
        print('sigma:', sigma, 'kernel:', kernel_size)
        self.CCN = net()
        self.gs = gs_layer(sigma=[sigma], kernel_size=kernel_size)
        self.gs = self.gs.cuda()
        
    @property
    def loss(self):
        return self.loss_1, self.loss_2*cfg.LAMBDA_1
    
    def forward(self, img, gt_map):
        pred = self.CCN(img)
        den_map = self.gs(gt_map)
        # print('CC:forward', img.shape, pred.shape, gt_map.shape, den_map.shape)
        # print('CC:forward', img.squeeze().shape, pred.squeeze().shape, gt_map.squeeze().shape, den_map.squeeze().shape)
        self.loss_1= self.loss_1_fn(pred.squeeze(), den_map.squeeze())
        self.loss_2= 1 - self.loss_2_fn(pred, den_map[:,None,:,:])
        return pred


    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map

