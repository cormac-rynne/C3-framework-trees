import torch
import torch.nn as nn
from misc import layer
from config import cfg
import models
import importlib.util
import os

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name,sigma=None, test=False):
        super(CrowdCounter, self).__init__()        

        # if model_name == 'AlexNet':
        #     from .SCC_Model.AlexNet import AlexNet as net
        # elif model_name == 'VGG':
        #     from .SCC_Model.VGG import VGG as net
        # elif model_name == 'VGG_DECODER':
        #     # if path is None:
        #     from .SCC_Model.VGG_decoder import VGG_decoder as net
        #     # else:
        #     #     fullpath = os.path.join(path, 'VGG_decoder.py')
        #     #     spec = importlib.util.spec_from_file_location('VGG_decoder', fullpath)
        #     #     foo = importlib.util.module_from_spec(spec)
        #     #     spec.loader.exec_module(foo)
        #     #     net = foo.VGG_decoder()
        # elif model_name == 'MCNN':
        #     from .SCC_Model.MCNN import MCNN as net
        # elif model_name == 'CSRNet':
        #     from .SCC_Model.CSRNet import CSRNet as net
        # elif model_name == 'Res50':
        #     from .SCC_Model.Res50 import Res50 as net
        # elif model_name == 'Res101':
        #     from .SCC_Model.Res101 import Res101 as net
        # elif model_name == 'Res101_SFCN':
        #     from .SCC_Model.Res101_SFCN import Res101_SFCN as net
        # elif model_name == 'SASNet':
        #     from .SCC_Model.sasnet import SASNet as net
        #
        # # experiments
        # elif model_name == 'arch11':
        #     from models.SCC_Model.innov.arch11 import arch11 as net
        # elif model_name == 'arch12':
        #     from models.SCC_Model.innov.arch12 import arch12 as net
        # elif model_name == 'arch13':
        #     from models.SCC_Model.innov.arch13 import arch13 as net
        # elif model_name == 'arch14':
        #     from models.SCC_Model.innov.arch14 import arch14 as net
        # elif model_name == 'arch15':
        #     from models.SCC_Model.innov.arch15 import arch15 as net
        # elif model_name == 'arch21':
        #     from models.SCC_Model.innov.arch21 import arch21 as net
        # elif model_name == 'arch22':
        #     from models.SCC_Model.innov.arch22 import arch22 as net
        # elif model_name == 'arch23':
        #     from models.SCC_Model.innov.arch23 import arch23 as net
        # elif model_name == 'arch24':
        #     from models.SCC_Model.innov.arch24 import arch24 as net
        # elif model_name == 'UNet':
        #     # if path:
        #     from .SCC_Model.UNet import UNet as net
        #     # else:
        #     #     fullpath = os.path.join(path, 'UNet.py')
        #     #     spec = importlib.util.spec_from_file_location('UNet', fullpath)
        #     #     foo = importlib.util.module_from_spec(spec)
        #     #     spec.loader.exec_module(foo)
        #     #     net = foo.UNet()

        net = getattr(models, model_name)
        gs_layer = getattr(layer, 'Gaussianlayer')

        if sigma is None:
            sigma = cfg.SIGMA
        kernel_size = 6*sigma+1
        print('sigma:',sigma, 'kernel:', kernel_size)

        if model_name == 'SASNet':
            self.CCN = net(block_size=32)
        else:
            self.CCN = net()

        self.gs = gs_layer(sigma=[sigma], kernel_size=kernel_size)

        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            self.gs = torch.nn.DataParallel(self.gs, device_ids=gpus).cuda()
        else:
            self.CCN=self.CCN.cuda()
            self.gs=self.gs.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, gt_map):
        pred = self.CCN(img)
        den_map = self.gs(gt_map)
        # print('CC:forward', img.shape, pred.shape, gt_map.shape, den_map.shape)
        self.loss_mse=self.build_loss(pred.squeeze(), den_map.squeeze())
        return pred
    
    def build_loss(self, density_map, gt_data):
        # print('CC:build loss', density_map.shape, gt_data.shape)
        loss_mse = self.loss_mse_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map
