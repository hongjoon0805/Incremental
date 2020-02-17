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
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        
        if self.args.cutmix:
            self.loss = trainer.CutMixCriterion('mean')
        else:
            self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self):
        
        self.train_data_iterator.dataset.update_exemplar()
        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()

    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False

    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        
        for data, target in tqdm(self.train_data_iterator):
            data = data.cuda()
            if self.args.cutmix:
                target1, target2, lamb = target
                target1, target2 = target1.cuda(), target2.cuda()
                target = (target1, target2, lamb)
            else:
                target = target.cuda()
            
            output = self.model(data)
            loss_CE = self.loss(output[:,:end], target)
            
            loss_KD = 0
            if tasknum > 0 and self.args.KD:
                T=2
                score = self.model_fixed(data).data
                loss_KD = torch.zeros(tasknum-1).cuda()
                for t in range(tasknum):
                    
                    # local distillation
                    KD_start = (t) * self.args.step_size
                    KD_end = (t+1) * self.args.step_size

                    soft_target = F.softmax(score[:,KD_start:KD_end] / T, dim=1)
                    output_log = F.log_softmax(output[:,KD_start:KD_end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target) * (T**2)
                
                loss_KD = loss_KD.sum()
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()