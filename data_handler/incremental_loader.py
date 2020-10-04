import copy
import logging
import time
import math

import numpy as np
import torch
import torch.utils.data as td
from sklearn.utils import shuffle
from PIL import Image

def CutMix(img, mix_img):
    
    # generate mixed sample
    lam = np.random.beta(1, 1)
    size = img.size()
    
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    img[:, bbx1:bbx2, bby1:bby2] = mix_img[:, bbx1:bbx2, bby1:bby2]
    
    return img


class IncrementalLoader(td.Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        
        self.t=0
        
        self.validation_buffer_size = int(self.args.memory_budget/10) * 2
        self.mode = None
        
        self.start = 0
        self.end = args.base_classes
        self.classes = self.dataset.classes
        
        self.train_start_idx = 0
        self.train_end_idx = np.argmax(self.dataset.train_labels>(self.end-1)) # end data index in training datset
        self.test_start_idx = 0
        self.test_end_idx = np.argmax(self.dataset.test_labels>(self.end-1)) # end data index in test dataset
        
        if self.end == self.classes:
            self.train_end_idx = len(self.dataset.train_labels)-1
            self.test_end_idx = len(self.dataset.test_labels)-1
        
        self.tr_idx = range(self.train_end_idx)
        self.eval_idx = range(self.train_end_idx)
        self.CutMix_idx = range(self.train_end_idx)
        
        self.memory_buffer = []
        self.exemplar = []
        self.start_point = []
        self.end_point = []
        for i in range(self.classes):
            self.start_point.append(np.argmin(self.dataset.train_labels<i))
            self.end_point.append(np.argmax(self.dataset.train_labels>(i)))
            self.memory_buffer.append([])
        self.end_point[-1] = len(self.dataset.train_labels)
    
    def get_cls_num_list(self):
        li = []
        # t > 0 인 상황에서만 사용한다고 가정
        for i in range(self.end):
            if i < self.start:
                li.append(len(self.memory_buffer[i]))
            else:
                li.append(self.end_point[i] - self.start_point[i])
        return li
    
    def task_change(self):
        self.t += 1
        
        self.start = self.end
        self.end += self.args.step_size
        
        self.train_start_idx = np.argmin(self.dataset.train_labels<self.start) # start data index
        self.train_end_idx = np.argmax(self.dataset.train_labels>(self.end-1)) # end data index
        self.test_end_idx = np.argmax(self.dataset.test_labels>(self.end-1)) # end data index
        
        if self.train_end_idx == 0:
            self.train_end_idx = self.dataset.train_labels.shape[0]
            self.test_end_idx = self.dataset.test_labels.shape[0]
        
        print(self.start, self.end, self.train_start_idx, self.train_end_idx)
        self.tr_idx = range(self.train_start_idx, self.train_end_idx)
        
        # validation set for bic
        if 'bic' in self.args.trainer and self.start < self.classes:
            val_per_class = (self.validation_buffer_size//2) // self.args.step_size
            self.tr_idx = []
            for i in range(self.args.step_size):
                end = self.end_point[self.start + i]
                start = self.start_point[self.start + i]
                self.validation_buffer += range(end-val_per_class, end)
                self.tr_idx += range(start, end-val_per_class)
        
        self.eval_idx = list(self.tr_idx) + self.exemplar
        factor = int(len(list(self.tr_idx)) / len(self.exemplar))
        self.CutMix_idx = list(self.tr_idx) + self.exemplar * factor
        if self.args.trainer != 'ssil' and self.args.trainer != 'lwf':
            self.tr_idx = list(self.tr_idx) + self.exemplar
            
    def update_bft_buffer(self):
        self.bft_buffer = copy.deepcopy(self.memory_buffer)
        min_len = 1e8
        for arr in self.bft_buffer:
            min_len = min(min_len, len(arr))

        buffer_per_class = min_len
        
        for i in range(self.start, self.end):
            start_idx = self.start_point[i]
            end_idx = self.end_point[i]
            idx = shuffle(np.arange(end_idx - start_idx), random_state = self.t)[:buffer_per_class]
            self.bft_buffer[i] += list(idx)
        for arr in self.bft_buffer:
            if len(arr) > buffer_per_class:
                arr.pop()

        self.bft_exemplar = []
        for arr in self.bft_buffer:
            self.bft_exemplar += arr
            
    def update_exemplar(self):
        
        buffer_per_class = math.ceil(self.args.memory_budget / self.end)
        # first, add new exemples

        for i in range(self.start,self.end):
            start_idx = self.start_point[i]
            self.memory_buffer[i] += range(start_idx, start_idx+buffer_per_class)
        # second, throw away the previous samples
        if buffer_per_class > 0:
            for i in range(self.start):
                if len(self.memory_buffer[i]) > buffer_per_class:
                    del self.memory_buffer[i][buffer_per_class:]

        # third, select classes from previous classes, and throw away only 1 samples per class
        # randomly select classes. **random seed = self.t or start** <-- IMPORTANT!

        length =sum([len(i) for i in self.memory_buffer])
        remain = length - self.args.memory_budget
        if remain > 0:
            imgs_per_class = [len(i) for i in self.memory_buffer]
            selected_classes = np.argsort(imgs_per_class)[-remain:]
            for c in selected_classes:
                self.memory_buffer[c].pop()

        self.exemplar = []
        for arr in self.memory_buffer:
            self.exemplar += arr
        
        
        # validation set for bic
        if 'bic' in self.args.trainer:
            self.bic_memory_buffer = copy.deepcopy(self.memory_buffer)
            self.validation_buffer = []
            validation_per_class = (self.validation_buffer_size//2) // self.end
            if validation_per_class > 0:
                for i in range(self.end):
                    self.validation_buffer += self.bic_memory_buffer[i][-validation_per_class:]
                    del self.bic_memory_buffer[i][-validation_per_class:]

            remain = self.validation_buffer_size//2 - validation_per_class * self.end

            if remain > 0:
                imgs_per_class = [len(i) for i in self.bic_memory_buffer]
                selected_classes = np.argsort(imgs_per_class)[-remain:]
                for c in selected_classes:
                    self.validation_buffer.append(self.bic_memory_buffer[c].pop())
            self.exemplar = []
            for arr in self.bic_memory_buffer:
                self.exemplar += arr
                
    def __len__(self):
        if self.mode == 'train':
            return len(self.tr_idx)
        elif self.mode == 'evaluate':
            return len(self.eval_idx)
        elif self.mode == 'CutMix':
            return len(self.CutMix_idx)
        elif self.mode == 'bias':
            return len(self.validation_buffer)
        elif self.mode == 'b-ft':
            return len(self.bft_exemplar)
        elif self.mode == 'full':
            return self.train_end_idx
        elif self.mode == 'test':
            return self.test_end_idx
    
    def __getitem__(self, index):
        data = self.dataset.train_data
        labels = self.dataset.train_labels
        transform = self.dataset.train_transform
        Mix = False
        if self.mode == 'train':
            index = self.tr_idx[index]
        elif self.mode == 'evaluate':
            index = self.eval_idx[index]
        elif self.mode == 'CutMix':
            index = self.CutMix_idx[index]
            label = labels[index]
            if label < self.end - self.args.step_size:
                mix_index = np.random.choice(self.memory_buffer[label])
                Mix = True
        elif self.mode == 'bias': # for bic bias correction
            index = self.validation_buffer[index]
        elif self.mode == 'b-ft':
            index = self.bft_exemplar[index]
        elif self.mode == 'full':
            pass
        elif self.mode == 'test':
            data = self.dataset.test_data
            labels = self.dataset.test_labels
            transform = self.dataset.test_transform
        
        img = data[index]
        
        try:
            img = Image.fromarray(img)
        except:
            img = Image.open(img, mode='r').convert('RGB')
        
        img = transform(img)
        
        if Mix:
            mix_img = data[mix_index]
            
            try:
                mix_img = Image.fromarray(mix_img)
            except:
                mix_img = Image.open(mix_img, mode='r').convert('RGB')
            
            mix_img = transform(mix_img)
            img = CutMix(img, mix_img)
            
        return img, labels[index]

class ResultLoader(td.Dataset):
    def __init__(self, dataset, args):
        
        self.t = 0
        
        self.args = args
        self.dataset = dataset
        self.classes = dataset.classes
        
        self.data = dataset.test_data
        self.labels = dataset.test_labels
        self.transform=dataset.test_transform
        
        
    def __len__(self):
        return len(self.test_idx)
    
    def reset(self):
        self.t = 0
        self.start = 0
        self.end = self.args.base_classes
        self.start_idx = 0
        self.end_idx = np.argmax(self.dataset.test_labels>(self.end-1)) # end data index in test dataset
        
        if self.end == self.classes:
            self.end_idx = len(self.dataset.test_labels)-1
            
        self.test_idx = range(self.start_idx, self.end_idx)
        
    def task_change(self):
        self.t += 1
        
        self.start = self.end
        self.end += self.args.step_size
        
        self.start_idx = np.argmin(self.labels<self.start) # start data index
        self.end_idx = np.argmax(self.labels>(self.end-1)) # end data index
        if self.end_idx == 0:
            self.end_idx = self.labels.shape[0]
        
        self.test_idx = range(self.start_idx, self.end_idx)
            
    def __getitem__(self, index):
        index = self.test_idx[index]
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = Image.open(img, mode='r').convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]