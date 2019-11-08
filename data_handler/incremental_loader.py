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
    def __init__(self, data, labels, classes, step_size, mem_sz, mode, batch_size, transform=None, base_classes=50):
        # label shuffle
        label_shffled = shuffle(np.arange(classes))
        labels = label_shuffled[labels]
        
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
        self.mem_sz = mem_sz
        self.mode=mode
        self.batch_size = batch_size
        self.start_idx = 0
        self.end_idx = self.len
        self.start = 0
        self.end = base_classes
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
        self.end_idx = self.end_idx  + self.step_size * self.data_per_classes
        self.start = self.end
        self.end += self.step_size
        self.len = self.data_par_classes * self.step_size
        
    def update_exemplar(self):
        j = 0
        print(len(self.exemplar))
        print(self.mem_sz)
        for idx in range(self.start_idx, self.end_idx):
            if len(self.exemplar) < self.mem_sz:
                self.exemplar.append(idx)
            else:
                i = np.random.randint(self.end+j)
                if i < self.mem_sz:
                    self.exemplar[i] = idx
            j += 1
                    
    
    def sample_exemplar(self):
#         exemplar_idx = shuffle(np.array(self.exemplar))[:self.batch_size]
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
            return self.end
    
    def __getitem__(self, index):
        '''
        Replacing this with a more efficient implemnetation selection; removing c
        :param index: 
        :return: 
        '''
        if self.mode == 'train':
            img = self.data[self.start_idx + index]
        else:
            img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'train':
            return img, self.labels[self.start_idx + index], self.labelsNormal[self.start_idx + index]
        else:
            return img, self.labels[index], self.labelsNormal[index]
