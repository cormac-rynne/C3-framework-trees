import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)

        if froze: self.frozePara()

    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps.unsqueeze(0))
        return gaussianmaps

    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

class Gaussianlayer(nn.Module):
    def __init__(self, sigma=[4], kernel_size=15):
        super(Gaussianlayer, self).__init__()
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, NL='relu'):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU() 
        else:
            self.relu = None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class convDU(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(9,1)
        ):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).resize(n,c,1,w)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)
            # pdb.set_trace()
            # fea[:,i,:,:] = self.conv(fea[:,i-1,:,:].expand(n,1,h,w))+fea[:,i,:,:].expand(n,1,h,w)


        for i in range(h):
            pos = h-i-1
            if pos == h-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]
        # pdb.set_trace()
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self,
        in_out_channels=2048,
        kernel_size=(1,9)
        ):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2)),
            nn.ReLU(inplace=True)
            )

    def forward(self, fea):
        n, c, h, w = fea.size()

        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).resize(n,c,h,1)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i-1])+i_fea)

        for i in range(w):
            pos = w-i-1
            if pos == w-1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos+1])+fea_stack[pos]


        fea = torch.cat(fea_stack, 3)
        return fea