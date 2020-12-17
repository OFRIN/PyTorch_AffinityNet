import os
import time
import torch
import random
import argparse
import numpy as np

from .txt_utils import add_txt

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_print(string, path):
    print(string)
    add_txt(path, string)

class AverageMeter:
    def __init__(self, keys):
        self.keys = keys
        self.data_dic = {key : [] for key in keys}
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = [np.mean(self.data_dic[key]) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]
            
        return dataset
    
    def clear(self):
        for key in self.data_dic.keys():
            self.data_dic[key] = []

class Timer:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.tik()
    
    def tik(self):
        self.start_time = time.time()
    
    def tok(self, ms = False, clear=False):
        self.end_time = time.time()
        
        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.tik()

        return duration