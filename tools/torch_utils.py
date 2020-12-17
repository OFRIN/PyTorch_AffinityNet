import cv2
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR

def flatten(x):
    return x.view(x.size(0), -1)

def global_average_pooling_2d(x, with_flatten=False):
    x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
    if with_flatten:
        x = flatten(x)
    return x

def one_hot_embedding(label, classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (int) class labels.
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

def calculate_parameters(model):
    return sum(param.numel() for param in model.parameters())/1000000.0

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def load_model(model, model_path, parallel=False):
    if parallel:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path, parallel=False):
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

def transfer_model(pretrained_model, model):
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()
    
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def get_cosine_schedule_with_warmup(optimizer,
                                    warmup_iteration,
                                    max_iteration,
                                    cycles=7./16.
                                    ):
    def _lr_lambda(current_iteration):
        if current_iteration < warmup_iteration:
            return float(current_iteration) / float(max(1, warmup_iteration))

        no_progress = float(current_iteration - warmup_iteration) / float(max(1, max_iteration - warmup_iteration))
        return max(0., math.cos(math.pi * cycles * no_progress))
    
    return LambdaLR(optimizer, _lr_lambda, -1)

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data