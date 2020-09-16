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
        
        #self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss = trainer.LSoftmaxLoss(2, model)
    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        if epoch == 0:  # and args.loss == 'lmargin'
            self.loss.beta = 0
            
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end - self.args.step_size
        loss_lsm = 0
        loss_cross = 0
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            
            output, feature = self.model(data, feature_return=True)
            
            CEloss = torch.nn.CrossEntropyLoss(reduction='mean')
            
            
            loss_CE = CEloss(output[:,:end], target)
            loss_cross += loss_CE
            
            loss_CE = self.loss(feature, target, end)
            

            loss_lsm += loss_CE
            
            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()
            
        print("beta %.6f"%self.loss.beta, "CE loss", loss_cross, "lsm", loss_lsm)