''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import copy
import logging

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms.functional as trnF

class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, batch_size, transform=None, loader = None, shuffle_idx=None, base_classes=50, strategy = 'Reservior', approach = 'coreset', self_sup = False):
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
        self.mode=mode
        self.batch_size = batch_size
        
        self.start = 0
        self.end = base_classes
        
        self.start_idx = 0
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        
        
        self.len = self.end_idx - self.start_idx
        self.current_len = self.len
        
        self.strategy = strategy
        self.approach = approach
        self.self_sup = self_sup
        self.exemplar = []
        self.start_point = []
        for i in range(classes):
            self.start_point.append(np.argmin(self.labelsNormal<i))
        
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b
        
    def task_change(self):
        self.t += 1
        
        self.start = self.end
        self.end += self.step_size
        
        self.start_idx = np.argmin(self.labelsNormal<self.start) # start data index
        self.end_idx = np.argmax(self.labelsNormal>(self.end-1)) # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]
        
        self.len = self.end_idx - self.start_idx
        self.current_len = self.len
        
        if self.approach == 'coreset':
            self.len += len(self.exemplar)
        
    def update_exemplar(self):
        
        if self.strategy == 'Reservior':
            self.Reservior()
        elif self.strategy == 'RingBuffer':
            self.RingBuffer()
        elif self.strategy == 'Weighted':
            self.Weighted()
            
#         exemplar_per_classes = np.zeros(self.total_classes)
#         for idx in self.exemplar:
#             exemplar_per_classes[idx//500] += 1
#         print(exemplar_per_classes)

    def Reservior(self):
        j = 0
        for idx in range(self.start_idx, self.end_idx):
            if len(self.exemplar) < self.mem_sz:
                self.exemplar.append(idx)
            else:
                i = np.random.randint(self.end_idx+j)
                if i < self.mem_sz:
                    self.exemplar[i] = idx
            j += 1
    
    def RingBuffer(self):
        buffer_per_class = self.mem_sz // self.end
        self.exemplar = []
        for i in range(self.end):
            start = self.start_point[i]
            self.exemplar += range(start,start+buffer_per_class)
    
    def Weighted(self):
        start = 0
        end = self.base_classes
        weight_sum = 0
        for t in reversed(range(1,self.t+2)):
            weight_sum += (end-start)*t
            start = end
            end += self.step_size
        
        k = self.mem_sz // weight_sum
        base = self.base_classes
        weight = self.t+1
        self.exemplar = []
        for i in range(self.end):
            start = self.start_point[i]
            if i> base:
                base += self.step_size
                weight = weight-1
            self.exemplar += range(start, start+weight*k)
    
    def __len__(self):
        if self.mode == 'train':
            return self.len
        else:
            return self.end_idx
    
    def __getitem__(self, index):
        
        if self.mode == 'train':
            if self.approach == 'coreset' and index >= self.current_len:
                index = self.exemplar[index - self.current_len]
            else:
                index = self.start_idx + index
                
        img = self.data[index]
        
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img[0])
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if self.transform is not None:
            img = self.transform(img)
            img = (img - mean) / std
            img = trnF.to_tensor(img.copy()).unsqueeze(0).numpy()
            img = np.concatenate((img, np.rot90(img, 1, axes=(2, 3)),
                            np.rot90(img, 2, axes=(2, 3)), np.rot90(img, 3, axes=(2, 3))), 0)
            img = torch.FloatTensor(img)

        return img, np.array([self.labelsNormal[index]]*4), np.array([0,1,2,3])

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
            img = self.loader(img[0])
            
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if self.transform is not None:
            img = self.transform(img)
            img = (img - mean) / std
            img = trnF.to_tensor(img.copy()).unsqueeze(0).numpy()
            img = np.concatenate((img, np.rot90(img, 1, axes=(2, 3)),
                            np.rot90(img, 2, axes=(2, 3)), np.rot90(img, 3, axes=(2, 3))), 0)
            img = torch.FloatTensor(img)

        return img, np.array([self.labelsNormal[index]]*4), np.array([0,1,2,3])

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