from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def compute_score(self, features, mean, precision, sqrt = False):
        
        batch_vec = (features.data.unsqueeze(1) - mean.unsqueeze(0))
        temp = torch.matmul(batch_vec, precision)
        if sqrt:
            out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).sqrt().squeeze()
        else:
            out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
        
        return out
    
    def train(self, epoch, mean = None, precision = None):
        
        T=0.125
        print('T: ', T)
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end-self.args.step_size
        start = 0
        lamb = mid / end
        
        if 'LDAM' in self.args.date:
            cls_num_list = np.ones(end)
            cls_num_list = self.incremental_loader.get_cls_num_list()
            self.loss = trainer.LDAMLoss(cls_num_list, s=1, max_m=self.args.margin)
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss_CE = self.loss(output[:,:end], target)
            
            loss_KD = 0
            
            if tasknum > 0:
                out, feature_prev = self.model_fixed(data, feature_return=True)
                score = self.compute_score(feature_prev, mean, precision, sqrt = True) * self.args.eta
                
                
#                 sample_num = 20
                
#                 print('Euclidean sqrt entropy')
#                 log_prob = F.log_softmax(score[:sample_num] / T, dim=1)
#                 prob = F.softmax(score[:sample_num] / T, dim=1)
#                 entropy = (-log_prob * prob).sum(dim=1)
#                 print(entropy)
                
#                 print('Euclidean sqrt softmax min & max')
#                 print(F.softmax(score[:sample_num] / T, dim=1).min(dim=1))
#                 print(F.softmax(score[:sample_num] / T, dim=1).max(dim=1))
                
#                 print('FC entropy')
#                 log_prob = F.log_softmax(out[:sample_num, :100] / T, dim=1)
#                 prob = F.softmax(out[:sample_num, :100] / T, dim=1)
#                 entropy = (-log_prob * prob).sum(dim=1)
#                 print(entropy)
                
#                 print('FC softmax min & max')
#                 print(F.softmax(out[:sample_num, :100] / T, dim=1).min(dim=1))
#                 print(F.softmax(out[:sample_num, :100] / T, dim=1).max(dim=1))
                
#                 print(woifdfngdklfj)

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:mid] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                        
            self.optimizer.zero_grad()
            (lamb * loss_KD + (1-lamb) * loss_CE).backward()
            self.optimizer.step()

