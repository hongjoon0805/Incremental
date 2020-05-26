import copy
import logging
import time
import cv2

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF

class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, batch_size, transform=None, loader = None, shuffle_idx=None, base_classes=50, strategy = 'Reservior', approach = 'bic'):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]
        
        sort_index = np.argsort(labels)
        self.data = data[sort_index]
        
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
        
        self.mem_sz = mem_sz
        self.bias_mem_sz = int(mem_sz/10)
        self.mode=mode
        self.batch_size = batch_size
        
        self.start = 0
        self.end = base_classes
        
        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        
        if self.end == classes:
            self.end_idx = len(labels)-1
        
        self.tr_idx = range(self.end_idx)
        self.len = self.end_idx - self.start_idx
        self.current_len = self.len
        
        self.strategy = strategy
        self.approach = approach
        self.exemplar = []
        self.bias_buffer = []
        self.start_point = []
        self.end_point = []
        for i in range(classes):
            self.start_point.append(np.argmin(self.labelsNormal<i))
            self.end_point.append(np.argmax(self.labelsNormal>(i)))
        self.end_point[-1] = len(labels)
        
        
    def task_change(self):
        self.t += 1
        
        self.start = self.end
        self.end += self.step_size
        
        self.start_idx = np.argmin(self.labelsNormal<self.start) # start data index
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]
        
        if self.approach == 'bic':
            val_per_class = self.bias_mem_sz // self.step_size
            self.tr_idx = []
            for i in range(self.step_size):
                end = self.end_point[self.start + i]
                start = self.start_point[self.start + i]
                self.bias_buffer += range(end-val_per_class, end)
                self.tr_idx += range(start, end-val_per_class)
        
        self.len = self.end_idx - self.start_idx
        if self.approach == 'bic':
            self.len -= self.bias_mem_sz
        self.current_len = self.len
        
        if self.approach == 'coreset' or self.approach == 'icarl' or self.approach == 'bic':
            self.len += len(self.exemplar)
        
    def update_exemplar(self):
        
        if self.strategy == 'RingBuffer':
            self.RingBuffer()
            
    def RingBuffer(self):
        buffer_per_class = self.mem_sz // self.end
        if self.approach == 'bic':
            buffer_per_class = int((self.mem_sz // self.end)*0.9)
            val_per_class = int((self.mem_sz // self.end)*0.1)
        self.exemplar = []
        self.bias_buffer = [] 
        for i in range(self.end):
            start = self.start_point[i]
            self.exemplar += range(start,start+buffer_per_class)
            if self.approach == 'bic':
                self.bias_buffer += range(start+buffer_per_class, start+buffer_per_class+val_per_class)
                
    def __len__(self):
        if self.mode == 'train':
            return self.len
        elif self.mode == 'bias':
            return len(self.bias_buffer)
        else:
            return self.end_idx
    
    def __getitem__(self, index):
        
        if self.mode == 'train':
            if (self.approach == 'coreset' or self.approach == 'icarl' or self.approach == 'bic') and index >= self.current_len:
                index = self.exemplar[index - self.current_len]
            else:
                if self.approach == 'bic':
                    index = self.tr_idx[index]
                else:
                    index = self.start_idx + index
        elif self.mode == 'bias':
            index = self.bias_buffer[index]
        img = self.data[index]
        
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]

class ResultLoader(td.Dataset):
    def __init__(self, data, labels, transform=None, loader = None):
        
        self.data = data
        self.labels = labels
        self.labelsNormal = np.copy(self.labels)
        self.transform=transform
        self.loader = loader
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img)
            
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]

def make_ResultLoaders(data, labels, classes, step_size, transform = None, loader = None, shuffle_idx=None, base_classes=50):
    if shuffle_idx is not None:
        labels = shuffle_idx[labels]
    sort_index = np.argsort(labels)
    data = data[sort_index]
    labels = np.array(labels)
    labels = labels[sort_index]
    
    start = 0
    end = base_classes
    
    loaders = []
    
    while(end <= classes):
        
        start_idx = np.argmin(labels<start) # start data index
        end_idx = np.argmax(labels>(end-1)) # end data index
        if end_idx == 0:
            end_idx = data.shape[0]
        
        loaders.append(ResultLoader(data[start_idx:end_idx], labels[start_idx:end_idx], transform=transform, loader=loader))
        
        start = end
        end += step_size
    
    return loaders