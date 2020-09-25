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
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end-self.args.step_size
        start = 0
        lamb = mid / end
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            
            if tasknum > 0:
                out, feature_prev = self.model_fixed(data, feature_return=True)
                score_normal = self.compute_score(feature_prev, mean, precision)
                score_sqrt = self.compute_score(feature_prev, mean, precision, sqrt = True)
                if 'min_max' in self.args.date:
                    min_val = score.min(dim=1, keepdim=True)[0]
                    max_val = score.max(dim=1, keepdim=True)[0]
                    max_min = max_val - min_val
                    
                    score = score / (max_min + 1e-11)
                
#                 sample_num = 1
#                 print('Euclidean normal score')
#                 print(score_normal[:sample_num])
                
#                 print('Euclidean sqrt score')
#                 print(score_sqrt[:sample_num])
                
#                 print('Euclidean normal softmax')
#                 print(F.softmax(score_normal[:sample_num] / T, dim=1))
                
#                 print('Euclidean normal softmax min & max')
#                 print(F.softmax(score_normal[:sample_num] / T, dim=1).min(dim=1))
#                 print(F.softmax(score_normal[:sample_num] / T, dim=1).max(dim=1))
                
#                 print('Euclidean sqrt softmax')
#                 print(F.softmax(score_sqrt[:sample_num] / T, dim=1))
                
#                 print('Euclidean sqrt softmax min & max')
#                 print(F.softmax(score_sqrt[:sample_num] / T, dim=1).min(dim=1))
#                 print(F.softmax(score_sqrt[:sample_num] / T, dim=1).max(dim=1))
                
#                 print('FC score')
#                 print(out[:sample_num,:100])
                
#                 print('FC softmax')
#                 print(F.softmax(out[:sample_num,:100] / T, dim=1))
                
#                 print(woifdfngdklfj)

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:mid] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                        
            self.optimizer.zero_grad()
            (lamb * loss_KD + (1-lamb) * loss_CE).backward()
            self.optimizer.step()

