from matplotlib import pyplot as plt

import matplotlib
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
import json
import cv2

import argparse
from misc import layer

from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import h5py
import re
from models.CC import CrowdCounter

ROOT_FOLDER = '/content/gdrive/My Drive/Documents/Masters/7CCSMPRJ - Individual Project/experiments/'
C3_EXP_FOLDER = os.path.join(os.getcwd(), 'exp')

STD_DCT = {
    'oilpalm': {
        'mean_std':(
            [0.4053, 0.4637, 0.3644],
            [0.1531, 0.1227, 0.1198]
        )
    },
    'blan': {
        'mean_std': (
            [0.5093, 0.5918, 0.4610],
            [0.1077, 0.0869, 0.0629]
        )
    },
    'neon': {
        'mean_std': (
            [0.5928, 0.5642, 0.4799],
            [0.1659, 0.1541, 0.1125]
        )
    }
}

def get_dataset(exp):
    pattern = "\d{1,2}-\d{1,2}_\d{1,2}-\d{1,2}_([^_]{1,20})_"
    dataset = re.findall(pattern, exp)[0]
    # print(f'Dataset: {dataset}')
    return dataset

def get_best_model(exp):
    model_lst = os.listdir(os.path.join(C3_EXP_FOLDER, exp))
    # print('Available:', model_lst)
    pattern = "ep_(\d{1,3})_"
    best = None
    best_ep = 0
    for model_ in model_lst:
        if not 'mae_' in model_:
            continue
        epoch = int(re.findall(pattern, model_)[0])
        if epoch > best_ep:
            best_ep = epoch
            best = model_
    model = best
    # print(f'Best: {model}')
    model_path = os.path.join(os.getcwd(), 'exp', exp, model)
    return model_path, model

def get_sigma(exp):
    pattern='_(\d{1,2})$'
    sigma=int(re.findall(pattern, exp)[0])
    return sigma

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Args
parser = argparse.ArgumentParser(description='c3')
parser.add_argument('--exp', '-e', metavar='EXPERIMENT', type=str, default=None,
                    help='exp name')
parser.add_argument('--model', '-m', metavar='MODEL', type=str, default=None,
                    help='model name')
args = parser.parse_args()
exp_folder = args.exp
dataset = get_dataset(exp_folder)
cfg.sigma = get_sigma(exp_folder)
sigma = cfg.sigma
if args.model:
    model = args.model
else:
    model_path, model = get_best_model(exp_folder)
print(model)


# Directories
data_folder = os.path.join(ROOT_FOLDER, 'data', dataset)
results_name = f'{exp_folder}__{model}'
exp_results_folder = os.path.join(ROOT_FOLDER, 'C-3-Framework', 'results', results_name)
exp_results_pred_folder = os.path.join(exp_results_folder, 'pred')
exp_results_gt_folder = os.path.join(exp_results_folder, 'gt')

os.makedirs(exp_results_folder, exist_ok=True)
os.makedirs(exp_results_pred_folder, exist_ok=True)
os.makedirs(exp_results_gt_folder, exist_ok=True)

mean_std = STD_DCT[dataset]['mean_std']

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

def main():

    file_list = []
    file_path = os.path.join(data_folder, 'test.json')
    with open(file_path, 'r') as f:
        file_list = json.load(f)
    test(file_list, model_path)

def test(file_list, model_path):

    net = CrowdCounter(cfg.GPU_ID,cfg.NET, sigma)
    state = torch.load(model_path)
    net.load_state_dict(state)
    net.cuda()
    net.eval()

    f1 = plt.figure(1)

    results_dct = {
        'filename': [],
        'gt': [],
        'pred': [],
        'diff': [],
        'filepath': []
    }

    for img_filepath in file_list:
        img_filename = os.path.split(img_filepath)[-1]
        _, extension = os.path.splitext(img_filename)
        filename_no_ext = os.path.splitext(img_filename)[0]
        print(img_filename)

        den_filepath = img_filepath.replace('img', 'gt_map').replace(extension, '.h5')
        with h5py.File(den_filepath, "r") as f:
            den = f['gt_map'][:]
        den = den.astype(np.float32, copy=False)

        # den = cv2.resize(den, (den.shape[1] // 8, den.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
        # den = Image.fromarray(den)
        # den = pd.read_csv(denname, sep=',',header=None).values
        # den = den.astype(np.float32, copy=False)

        img = Image.open(img_filepath)
        if img.mode == 'L':
            img = img.convert('RGB')
        img = img_transform(img)

        gt_count = np.round(np.sum(den))
        with torch.no_grad():
            img = Variable(img[None,:,:,:]).cuda()
            pred_map = net.test_forward(img)
            den = Variable(den[None, :, :, :]).cuda()
            den = net.gs(den)

        # sio.savemat(exp_name+'/pred/'+filename_no_ext+'.mat',{'data':pred_map.squeeze().cpu().numpy()/100.})
        # sio.savemat(exp_name+'/gt/'+filename_no_ext+'.mat',{'data':den})

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
        pred_count = np.round(np.sum(pred_map)/100.0)
        pred_map = pred_map/np.max(pred_map+1e-20)
        
        den = den/np.max(den+1e-20)
        diff = den - pred_map

        results_dct['filename'].append(img_filename)
        results_dct['gt'].append(int(gt_count))
        results_dct['diff'].append(int(gt_count - pred_count))
        results_dct['pred'].append(int(pred_count))
        results_dct['filepath'].append(img_filepath)

        # den_frame = plt.gca()
        # plt.imshow(den, 'jet')
        # den_frame.axes.get_yaxis().set_visible(False)
        # den_frame.axes.get_xaxis().set_visible(False)
        # den_frame.spines['top'].set_visible(False)
        # den_frame.spines['bottom'].set_visible(False)
        # den_frame.spines['left'].set_visible(False)
        # den_frame.spines['right'].set_visible(False)
        # filename = filename_no_ext+'_gt_'+str(int(gt_count))+'.png'
        # filepath = os.path.join(exp_results_folder, filename)
        # plt.savefig(filepath, bbox_inches='tight',pad_inches=0,dpi=150)
        # plt.close()
        #
        # # sio.savemat(exp_name+'/'+filename_no_ext+'_gt_'+str(int(gt))+'.mat',{'data':den})
        #
        # pred_frame = plt.gca()
        # plt.imshow(pred_map, 'jet')
        # pred_frame.axes.get_yaxis().set_visible(False)
        # pred_frame.axes.get_xaxis().set_visible(False)
        # pred_frame.spines['top'].set_visible(False)
        # pred_frame.spines['bottom'].set_visible(False)
        # pred_frame.spines['left'].set_visible(False)
        # pred_frame.spines['right'].set_visible(False)
        # filename = filename_no_ext+'_pred_'+str(float(pred_count))+'.png'
        # filepath = os.path.join(exp_results_folder, filename)
        # plt.savefig(filepath, bbox_inches='tight',pad_inches=0,dpi=150)
        # plt.close()
        #
        # # sio.savemat(exp_name+'/'+filename_no_ext+'_pred_'+str(float(pred))+'.mat',{'data':pred_map})
        #
        # diff_frame = plt.gca()
        # plt.imshow(diff, 'jet')
        # plt.colorbar()
        # diff_frame.axes.get_yaxis().set_visible(False)
        # diff_frame.axes.get_xaxis().set_visible(False)
        # diff_frame.spines['top'].set_visible(False)
        # diff_frame.spines['bottom'].set_visible(False)
        # diff_frame.spines['left'].set_visible(False)
        # diff_frame.spines['right'].set_visible(False)
        # filename = filename_no_ext+'_diff.png'
        # filepath = os.path.join(exp_results_folder, filename)
        # plt.savefig(filepath, bbox_inches='tight',pad_inches=0,dpi=150)
        #
        # plt.close()

        # sio.savemat(exp_name+'/'+filename_no_ext+'_diff.mat',{'data':diff})

    filepath = os.path.join(exp_results_folder, 'data.json')
    with open(filepath, 'w') as f:
        json.dump(results_dct, f, indent=4)

if __name__ == '__main__':
    main()




