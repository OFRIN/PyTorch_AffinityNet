import os
import cv2
import glob
import torch

import numpy as np

from PIL import Image

from .xml_utils import *
from .torch_utils import one_hot_embedding

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, txt_path, class_dic, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        
        self.class_dic = class_dic
        self.classes = len(self.class_dic.keys())

        self.transform = transform

        self.image_names = [image_name.strip() + '.jpg' for image_name in open(txt_path).readlines()]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        # preprocessing
        image_name = self.image_names[index]
        _, tags = read_xml(self.xml_dir + image_name.replace('.jpg', '.xml'))

        # open image and transform
        image = Image.open(self.image_dir + image_name)

        if len(np.shape(image)) == 2:
            image = image.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        # collect shape and tags
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)

        return image, label

class VOC_Dataset_with_Mask(torch.utils.data.Dataset):
    def __init__(self, root_dir, txt_path, class_dic, transform=None):
        self.root_dir = root_dir
        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'
        
        self.class_dic = class_dic
        self.classes = len(self.class_dic.keys())

        self.transform = transform

        self.image_names = [image_name.strip() + '.jpg' for image_name in open(txt_path).readlines()]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        # preprocessing
        image_id = self.image_names[index].replace('.jpg', '')
        _, tags = read_xml(self.xml_dir + image_id + '.xml')

        # open image and transform
        image = Image.open(self.image_dir + image_id + '.jpg')

        if len(np.shape(image)) == 2:
            image = image.convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)

        # collect shape and tags
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        
        mask = cv2.imread(self.mask_dir + image_id + '.png')
        
        return image_id, image, label, mask

def color_map(N = 256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b, g, r])

    return cmap

def get_color_map_dic(option):
    if option == 'PASCAL_VOC':
        labels = ['background', 
                  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']

        n_classes = 21
        h = 20
        w = 500

        color_index_list = [index for index in range(n_classes)]
        color_index_list.append(-1)

        cmap = color_map()
        cmap_dic = {label : cmap[color_index] for label, color_index in zip(labels, range(n_classes))}
        cmap_image = np.empty((h * len(labels), w, 3), dtype = np.uint8)

        for color_index in color_index_list:
            cmap_image[color_index * h : (color_index + 1) * h, :] = cmap[color_index]
    
    return cmap_dic, cmap_image