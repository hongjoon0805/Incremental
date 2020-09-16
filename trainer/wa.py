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

    def train(self, epoch):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        start = end-self.args.step_size
        
        lamb = start / end
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            if tasknum > 0:
                end_KD = start
                start_KD = end_KD - self.args.step_size
                
                score = self.model_fixed(data)[:,:end_KD].data
                
                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                
            self.optimizer.zero_grad()
            (lamb*loss_KD + (1-lamb)*loss_CE).backward()
            self.optimizer.step()
            
            self.model.module.fc.bias.data[:] = 0
            
            # weight cliping 0인걸 없애기
            weight = self.model.module.fc.weight.data
            #print(weight.shape)
            weight[weight < 0] = 0

        #for p in self.model.module.fc.weight:
            #print(p)
            #print((p==0).sum())

    def weight_align(self):
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        weight = self.model.module.fc.weight.data
        
        prev = weight[:start, :]
        new = weight[start:end, :]
        print(prev.shape, new.shape)
        mean_prev = torch.mean(torch.norm(prev, dim=1)).item()
        mean_new = torch.mean(torch.norm(new, dim=1)).item()

        gamma = mean_prev/mean_new
        print(mean_prev, mean_new, gamma)
        new = new * gamma
        result = torch.cat((prev, new), dim=0)
        weight[:end, :] = result
        print(torch.mean(torch.norm(self.model.module.fc.weight.data[:start], dim=1)).item())
        print(torch.mean(torch.norm(self.model.module.fc.weight.data[start:end], dim=1)).item())
