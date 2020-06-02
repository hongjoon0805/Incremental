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
import torch.nn as nn
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm

import networks
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        
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
        start = 0
        end = self.train_data_iterator.dataset.end
        mid = end-self.args.step_size
        kwargs = {'num_workers': self.args.workers, 'pin_memory': True}
        
        exemplar_dataset_loaders = trainer.ExemplarLoader(self.train_data_iterator.dataset)
        exemplar_iterator = torch.utils.data.DataLoader(exemplar_dataset_loaders,
                                                        batch_size=self.args.replay_batch_size, 
                                                        shuffle=True, drop_last=True, **kwargs)
        
        
        if tasknum > 0:
            iterator = zip(self.train_data_iterator, exemplar_iterator)
        else:
            iterator = self.train_data_iterator
            
        for samples in tqdm(iterator):
            if tasknum > 0:
                curr, prev = samples
                
                data, target = curr
                if self.args.ablation == 'None':
                    target = target%(end-mid)
                batch_size = data.shape[0]
                data_r, target_r = prev
                replay_size = data_r.shape[0]
                data, data_r = data.cuda(), data_r.cuda()
                data = torch.cat((data,data_r))
                target, target_r = target.cuda(), target_r.cuda()
                
            else:
                data, target = samples
                data = data.cuda()
                target = target.cuda()
                    
                batch_size = data.shape[0]
            
            output = self.model(data)
            
            if self.args.ablation == 'naive':
                target = torch.cat((target, target_r))
                loss_CE = self.loss(output[:,:end],target) / (batch_size + replay_size)
            
            else:
            
                loss_CE_curr = 0
                loss_CE_prev = 0

                curr = output[:batch_size,mid:end]
                loss_CE_curr = self.loss(curr, target)

                if tasknum > 0:
                    prev = output[batch_size:batch_size+replay_size,start:mid]
                    loss_CE_prev = self.loss(prev, target_r)
                    loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

                else:
                    loss_CE = loss_CE_curr / batch_size
                
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
