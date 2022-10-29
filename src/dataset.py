import os
import glob

import PIL
import cv2
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
# from scipy.misc import imread
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        self.mode=config.MODE

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img = imread(self.data[index])
        # load edge
        edge = self.load_edge(img, index)

        if(self.mode==1):
            # random crop
            if(size< img.shape[0]):
                length = random.randint(size, img.shape[0])
                pad = img.shape[0] - length
                nh = random.randint(0, pad)
                nw = random.randint(0, pad)
                img = img[nh:nh + length, nw:nw + length]
                edge = edge[nh:nh + length, nw:nw + length]


        if(img.shape[0]==img.shape[1]):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            edge = cv2.resize(edge, (size, size), interpolation=cv2.INTER_AREA)

        # else:
        #     img = cv2.resize(img, (int(img.shape[0]) , int(img.shape[1])), interpolation=cv2.INTER_AREA)


       # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        # if size != 0:
        #     img = self.resize(img, size, size)

        # img = self.resize(img, 512, 512)
        # create grayscale image
        img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)


        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        # mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        # if self.edge == 1:
            # # no edge
            # if sigma == -1:
            #     return np.zeros(img.shape).astype(np.float)
            #
            # # random sigma
            # if sigma == 0:
            #     sigma = random.randint(1, 4)
            #
            # return canny(img, sigma=sigma, mask=mask).astype(np.float)

        # external
        # else:
        imgh, imgw = img.shape[0:2]
        edge = imread(self.edge_data[index])
        edge = rgb2gray(edge)
        edge = 1 - edge
        edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            # if self.nms == 1:
            #     edge = edge * canny(img, sigma=sigma, mask=mask)

        return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        # img = img.copy()
        img_t = F.to_tensor(img).float()
        # print(img_t.shape)
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = Image.fromarray(img)
        img = np.array(img.resize([height, width], PIL.Image.BICUBIC))

        # img = scipy.misc.imresize(img, [height, width])

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
