from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end - self.args.step_size
        
        if 'Hinge' in self.args.date:
            cls_num_list = np.ones(end)
            self.loss = trainer.LDAMLoss(cls_num_list, max_m=1, s=1, mode='Hinge')
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            
            if tasknum > 0 and self.args.prev_new:
                loss_CE_curr = 0
                loss_CE_prev = 0
                curr_mask = target >= mid
                prev_mask = target < mid
                curr_num = (curr_mask).sum().int()
                prev_num = (prev_mask).sum().int()
                batch_size = curr_num + prev_num
                
                loss_CE_curr = self.loss(output[curr_mask,mid:end], target[curr_mask]%(end-mid)) * curr_num
                loss_CE_prev = 0
                if prev_num > 0:
                    loss_CE_prev = self.loss(output[prev_mask,:mid], target[prev_mask]) * prev_num
                loss_CE = (loss_CE_curr + loss_CE_prev) / batch_size

            else:
                loss_CE = self.loss(output[:,:end], target)
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
            
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

