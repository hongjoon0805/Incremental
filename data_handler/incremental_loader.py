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


class IncrementalLoader(td.Dataset):
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, batch_size, transform=None, shuffle_idx=None, base_classes=50, strategy = 'Reservior', approach = 'coreset'):
        if shuffle_idx is not None:
            # label shuffle
            print("Label shuffled")
            labels = shuffle_idx[labels]
            print(shuffle_idx)
        
        sort_index = np.argsort(labels)
        if "torch" in str(type(data)):
            data = data.numpy()
        self.data = data[sort_index]
        labels = np.array(labels)
        self.labels = labels[sort_index]
        self.labelsNormal = np.copy(self.labels)
        self.transform = transform
        self.total_classes = classes
        self.data_per_classes = data.shape[0] // classes
        self.step_size = step_size
        self.base_classes = base_classes
        self.t=0
        self.len = self.data_per_classes * base_classes
        self.current_len = self.len
        self.mem_sz = mem_sz
        self.mode=mode
        self.batch_size = batch_size
        self.start_idx = 0
        self.end_idx = self.len
        self.start = 0
        self.end = base_classes
        
        self.strategy = 'Reservior'
        self.approach = approach
        self.exemplar = []
        
        self.transformLabels()

    def transformLabels(self):
        '''Change labels to one hot coded vectors'''
        b = np.zeros((self.labels.size, self.labels.max() + 1))
        b[np.arange(self.labels.size), self.labels] = 1
        self.labels = b
        
    def task_change(self):
        self.t += 1
        self.start_idx = self.end_idx 
        self.end_idx += self.step_size * self.data_per_classes
        self.start = self.end
        self.end += self.step_size
        self.len = self.data_per_classes * self.step_size
        if self.approach == 'coreset':
            self.len = self.data_per_classes * self.step_size + len(self.exemplar)
        
    def update_exemplar(self):
        
        if self.strategy == 'Reservior':
            self.Reservior()
        elif self.strategy == 'RingBufferr':
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
            
            self.exemplar += range(i*self.data_per_classes,i*self.data_per_classes+buffer_per_class)
    
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
            if i> base:
                base += self.step_size
                weight = weight-1
            self.exemplar += range(i*self.data_per_classes, i*self.data_per_classes+weight*k)
    
    def sample_exemplar(self):
        exemplar_idx = shuffle(np.array(self.exemplar))[:self.batch_size]
        
        img_arr = []
        labels_arr = []
        labels_Normal_arr = []
        
        for idx in exemplar_idx:
            img = self.data[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            img_arr.append(img)
            labels_arr.append(torch.tensor(self.labels[idx]))
            labels_Normal_arr.append(torch.tensor(self.labelsNormal[idx]))
        
        img = torch.stack(img_arr)
        labels = torch.stack(labels_arr)
        labelsNormal = torch.stack(labels_Normal_arr)
        
        return img, labels, labelsNormal
        
    
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
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index], self.labelsNormal[index]
        
#     def __getitem__(self, index):
        
#         if self.mode == 'train':
#             img = self.data[self.start_idx + index]
#         else:
#             img = self.data[index]
#         img = Image.fromarray(img)
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.mode == 'train':
#             return img, self.labels[self.start_idx + index], self.labelsNormal[self.start_idx + index]
#         else:
#             return img, self.labels[index], self.labelsNormal[index]
