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

    def increment_classes(self, t):
         
        self.train_data_iterator.dataset.task_change(t)
        self.test_data_iterator.dataset.task_change(t)

    def setup_training(self, lr):
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr

    def update_frozen_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.module.fc.parameters():
            param.requires_grad = True

    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        mid = end - self.args.step_size
        
        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.cuda()
            
            if( target >= (tasknum+1)*self.args.step_size).sum() > 0:
                print("error")

            output = self.model(data)
            
            loss_CE = self.loss(output[:,:end], target)
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
