import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys
import random
from PIL import Image
from torchvision.utils import make_grid
import cv2

def normalize(data):
    return data / 255.

def crop(imgA, imgB, patch_size):

    if imgA.shape != imgB.shape:
        print(f"imgA.shape != imgB.shape, {imgA.shape} != {imgB.shape}")
        raise StopIteration
    h, w, c = imgA.shape
    patch_size = patch_size[0]

    r = random.randrange(0, h - patch_size + 1)
    c = random.randrange(0, w - patch_size + 1)

    O = imgA[r: r + patch_size, c: c + patch_size]  # rain
    B = imgB[r: r + patch_size, c: c + patch_size]  # norain


    return O, B

def augment(imgA, imgB, hflip=True, rot=True):

    h = hflip and random.random() < 0.5
    v = rot and random.random() < 0.5
    if h:
            imgA = np.flip(imgA, axis=1).copy()
            imgB = np.flip(imgB, axis=1).copy()
    if v:
            imgA = np.flip(imgA, axis=0).copy()
            imgB = np.flip(imgB, axis=0).copy()
    return imgA, imgB


class RainDS_Dataset(data.Dataset):
    def __init__(self, path, patch_size, eval=False, format='.png', dataset_type='all'):
        super(RainDS_Dataset, self).__init__()
        self.patch_size = patch_size
        # print('crop size',size)
        self.eval = eval

        self.format = format
        self.dataset_type = dataset_type

        dir_tmp = 'train' if not self.eval else 'test'

        self.gt_path = os.path.join(path, dir_tmp, 'gt')

        self.gt_list = []
        self.rain_list = []

        raindrop_path = os.path.join(path, dir_tmp, 'raindrop')
        rainstreak_path = os.path.join(path, dir_tmp, 'rainstreak')
        streak_drop_path = os.path.join(path, dir_tmp, 'rainstreak_raindrop')

        raindrop_names = os.listdir(raindrop_path)
        rainstreak_names = os.listdir(rainstreak_path)
        streak_drop_names = os.listdir(streak_drop_path)

        rd_input = []
        rd_gt = []

        rs_input = []
        rs_gt = []

        rd_rs_input = []
        rd_rs_gt = []

        for name in raindrop_names:
            rd_input.append(os.path.join(raindrop_path, name))
            gt_name = name.replace('rd', 'norain')
            rd_gt.append(os.path.join(self.gt_path, gt_name))

        for name in rainstreak_names:
            rs_input.append(os.path.join(rainstreak_path, name))
            gt_name = name.replace('rain', 'norain')
            rs_gt.append(os.path.join(self.gt_path, gt_name))

        for name in streak_drop_names:
            rd_rs_input.append(os.path.join(streak_drop_path, name))
            gt_name = name.replace('rd-rain', 'norain')
            rd_rs_gt.append(os.path.join(self.gt_path, gt_name))

        if dataset_type == 'all':
            self.gt_list += rd_gt
            self.rain_list += rd_input
            self.gt_list += rs_gt
            self.rain_list += rs_input
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input
        elif dataset_type == 'rs':
            self.gt_list += rs_gt
            self.rain_list += rs_input
        elif dataset_type == 'rd':
            self.gt_list += rd_gt
            self.rain_list += rd_input
        elif dataset_type == 'rsrd':
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input



    def __getitem__(self, index):
        input_file = self.rain_list[index]
        target_file = self.gt_list[index]
        input_img = cv2.imread(input_file)
        target = cv2.imread(target_file)
        target_img = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        b, g, r = cv2.split(input_img)
        input_img = cv2.merge([r, g, b])
        imgB = np.float32(normalize(target_img))
        imgA = np.float32(normalize(input_img))
        input_file = os.path.basename(input_file)
        target_file = os.path.basename(target_file)
        if not self.eval:
            imgA, imgB = crop(imgA, imgB, patch_size=self.patch_size)
            imgA, imgB = augment(imgA, imgB)
        else:
            print(input_file, target_file)#, imgA.shape, imgB.shape)
        imgA = np.transpose(imgA, (2, 0, 1))  # rain
        imgB = np.transpose(imgB, (2, 0, 1))
        return {'O': imgA, 'B': imgB, 'filename': os.path.join(self.dataset_type, target_file[:-4])}

    def __len__(self):

        return len(self.rain_list)