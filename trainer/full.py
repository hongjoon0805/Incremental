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
        self.incremental_loader.mode = 'full'
        self.train_iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, drop_last=True, 
                                                          shuffle=True, **self.kwargs)
        
    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
            
        self.model = networks.ModelFactory.get_model(self.args.dataset)
        self.model = torch.nn.DataParallel(self.model).cuda()
    
    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        incremental_loader.mode = 'full'
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            loss_CE = self.loss(output[:,:end], target)
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
