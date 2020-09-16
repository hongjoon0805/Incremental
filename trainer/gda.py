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
        
    def compute_score(self, features, mean, precision):
        
        batch_vec = (features.data.unsqueeze(1) - mean.unsqueeze(0))
        temp = torch.matmul(batch_vec, precision)
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
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            
            if tasknum > 0:
                _, feature_prev = self.model_fixed(data, feature_return=True)
                score = self.compute_score(feature_prev, mean, precision)

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:mid] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                        
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()

