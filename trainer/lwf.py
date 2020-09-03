''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks

class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, model, args, optimizer):
        super().__init__(trainDataIterator, model, args, optimizer)

    def train(self, epoch):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.cuda()
            
        
            output = self.model(data)
            loss_CE = self.loss(output[:,start:end], target[:,start:end])
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data).data
                loss_KD = torch.zeros(tasknum).cuda()
                for t in range(tasknum):
                    
                    # local distillation
                    start = (t) * self.args.step_size
                    end = (t+1) * self.args.step_size

                    soft_target = F.softmax(score[:,start:end] / T, dim=1)
                    output_log = F.log_softmax(output[:,start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target) * (T**2) * self.args.alpha
                
                loss_KD = loss_KD.sum()
                
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))

