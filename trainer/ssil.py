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
    def __init__(self, IncrementalLoader, model, args):
        super().__init__(IncrementalLoader, model, args)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        
    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end-self.args.step_size
        start = 0
        
        exemplar_dataset_loaders = trainer.ExemplarLoader(self.incremental_loader)
        exemplar_iterator = torch.utils.data.DataLoader(exemplar_dataset_loaders,
                                                        batch_size=self.args.replay_batch_size, 
                                                        shuffle=True, drop_last=True, **self.kwargs)
        
        if tasknum > 0:
            iterator = zip(self.train_iterator, exemplar_iterator)
        else:
            iterator = self.train_iterator
        
        self.incremental_loader.mode = 'train'
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
