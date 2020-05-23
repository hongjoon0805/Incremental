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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    def forward(self, x, start, end):
        x[:,start:end] *= self.alpha
        x[:,start:end] += self.beta
        return x

class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bias_correction_layer = BiasLayer()
        
        self.bias_optimizer = torch.optim.SGD(self.bias_correction_layer.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]
                    
    def update_bias_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp]*2 == epoch:
                for param_group in self.optimizer.bias_optimizer:
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

    def get_model(self):
        myModel = networks.ModelFactory.get_model(self.args.dataset).cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def train(self, epoch):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        
        lamb = start / end
        
        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data)[:,:end].data
                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                
            self.optimizer.zero_grad()
            (lamb*loss_KD + (1-lamb)*loss_CE).backward()
            self.optimizer.step()
            
    def train_bias_correction(self, bias_iterator):
        # bias train iterator가 필요하다 validation data 2000장 정도 필요함. (1000 + 1000)
        # output 뽑고 new 쪽에만 forward 하도록 짜야한다.
        # bias correction layer 전용 optimizer가 필요하다.
        
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        
        for data, target in tqdm(bias_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            output = self.bias_correction_layer(output, start, end)
            
            loss_CE = self.loss(output, target)
            
            self.bias_optimizer.zero_grad()
            (loss_CE).backward()
            self.bias_optimizer.step()
        