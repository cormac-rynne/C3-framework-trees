import numpy as np
import os
import random
from scipy import io as sio
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
import h5py
import cv2
import json

import pandas as pd

from config import cfg


class Blan(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.data_files = None
        if mode == 'train':
            filepath = os.path.join(data_path, 'train.json')
            # print(filepath)
        elif mode == 'val':
            filepath = os.path.join(data_path, 'val.json')
            # print(filepath)
        with open(filepath, 'r') as f:
            self.data_files = json.load(f)
        if self.data_files is None:
            raise ValueError(f'No data for mode {mode}')
        self.num_samples = len(self.data_files)

    def __getitem__(self, index):
        img_filepath = self.data_files[index]

        img, den = self.read_image_and_gt(img_filepath)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        # print(fname, img.shape, den.shape)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, img_filepath):
        img = Image.open(img_filepath)
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'),
        #                   sep=',', header=None, index_col=None).values
        den_filepath = img_filepath.replace('.tif', '.h5').replace('img', 'gt_map')
        with h5py.File(den_filepath, "r") as f:
            den = f['gt_map'][:]

        den = den.astype(np.float32, copy=False)

        # den = cv2.resize(den, (den.shape[1] // 8, den.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64
        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples       
            
        