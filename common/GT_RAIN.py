import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
# from skimage.measure.s/imple_metrics import compare_psnr
import re

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch




def normalize(data):
    return data / 255.



def crop(imgA, imgB, patch_size):
    # patch_size = .patch_size
    if imgA.shape != imgB.shape:
        print(f"imgA.shape != imgB.shape, {imgA.shape} != {imgB.shape}")
        raise StopIteration
    h, w, c = imgA.shape
    patch_size = patch_size[0]
    # if aug:
    #     mini = - 1 / 4 * patch_size
    #     maxi = 1 / 4 * patch_size + 1
    #     p_h = patch_size + self.rand_state.randint(mini, maxi)
    #     p_w = patch_size + self.rand_state.randint(mini, maxi)
    # else:
    #     p_h, p_w = patch_size, patch_size
    #
    # r = self.rand_state.randint(0, h - p_h)
    # c = self.rand_state.randint(0, w - p_w)
    r = random.randrange(0, h - patch_size + 1)
    c = random.randrange(0, w - patch_size + 1)

    # O = img_pair[:, w:]
    # B = img_pair[:, :w]
    O = imgA[r: r + patch_size, c: c + patch_size]  # rain
    B = imgB[r: r + patch_size, c: c + patch_size]  # norain
    # cv2.imshow("O", O)
    # cv2.imshow("B", B)
    # cv2.waitKey(1000)

    # if aug:
    #     O = cv2.resize(O, (patch_size, patch_size))
    #     B = cv2.resize(B, (patch_size, patch_size))

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

class GTRAIN_Dataset(udata.Dataset):
    def __init__(self, data_path, patch_size, eval):
        super(GTRAIN_Dataset, self).__init__()
        # train
        print('process training data')
        self.input_path = os.path.join(data_path, 'input')
        self.target_path = os.path.join(data_path, 'gt')
        self.patch_size = patch_size
        self.imgs = os.listdir(self.input_path)
        self.eval = eval
        self.offset_list = []
        self.index_list = []
        self.num = len(self.imgs)

        print(f'{data_path}, # samples {self.num}\n')

    def __getitem__(self, idx):

        input_file = self.imgs[idx % self.num]
        target_file = input_file

        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        target = cv2.imread(os.path.join(self.target_path, target_file))
        print(target_file)
        print(target.shape)
        target_img = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        b, g, r = cv2.split(input_img)
        input_img = cv2.merge([r, g, b])
        imgB = np.float32(normalize(target_img))
        imgA = np.float32(normalize(input_img))
        if not self.eval:
            imgA, imgB = crop(imgA, imgB, patch_size=self.patch_size)
            imgA, imgB = augment(imgA, imgB)
        else:
            print(input_file, target_file)#, imgA.shape, imgB.shape)
        imgA = np.transpose(imgA, (2, 0, 1))  # rain
        imgB = np.transpose(imgB, (2, 0, 1))

        return {'O': imgA, 'B': imgB, 'filename': input_file[:-4]}
        # return imgA, input_file[:-4], target_file[:-4]

    def __len__(self):

        return self.num





if __name__ == '__main__':

    data_path = "./data/LHP/train"
    dataset = GTRAIN_Dataset(data_path, (128, 128), False)
    print(dataset[0])
