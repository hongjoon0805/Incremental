import copy
import logging
import time
import math

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF

class FullLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, transform=None, loader = None, shuffle_idx=None, base_classes=50, approach = 'bic', model=None):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]
        
        sort_index = np.argsort(labels)
        self.data = data[sort_index]
        
        self.model = model
        
        labels = np.array(labels)
        self.labels = labels[sort_index]
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.loader = loader
        self.total_classes = classes
        
        # Imagenet에서는 class shuffle 후 label < current_class 에서 argmin을 찾으면 length 출력 가능하다.
        
        self.step_size = step_size
        self.base_classes = base_classes
        self.t=0
        
        self.start = 0
        self.end = base_classes

        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        
        if self.end == classes:
            self.end_idx = len(labels)-1
        
        self.tr_idx = range(self.end_idx)
        self.len = len(self.tr_idx)
        
    def task_change(self, t=0):
        self.t = t
        
        if t == 0:
            self.end = self.base_classes
        else:    
            self.end = self.base_classes + self.step_size * t
        
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]
        
        self.tr_idx = range(self.start_idx, self.end_idx)
        
        self.len = len(self.tr_idx)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
#         time.sleep(0.1)
        index = self.tr_idx[index]
        
        img = self.data[index]
        
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]

        
