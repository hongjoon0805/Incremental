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
        
    
    def balance_fine_tune(self):
        self.update_frozen_model()
        self.setup_training(self.args.bft_lr)
        
        self.incremental_loader.update_bft_buffer()
        self.incremental_loader.mode = 'b-ft'
        
        schedule = np.array(self.args.schedule)
        bftepoch = int(self.args.nepochs*3/4)
        for epoch in range(bftepoch):
            self.update_lr(epoch, schedule)
            self.train(epoch, bft=True)
        
    def train(self, epoch, bft=False):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.incremental_loader.t
        end = self.incremental_loader.end
        start = end-self.args.step_size
        
        
        if 'LDAM' in self.args.date:
            cls_num_list = np.ones(end)
            cls_num_list = self.incremental_loader.get_cls_num_list()
            self.loss = trainer.LDAMLoss(cls_num_list, s=1, max_m=self.args.margin)
        
        for data, target in tqdm(self.train_iterator):
            data, target = data.cuda(), target.cuda()
            try:
                output = self.model(data)[:,:end]
            except:
                continue
            if bft:
                loss_CE = torch.nn.CrossEntropyLoss(reduction='mean')(output, target)
            else:
                loss_CE = self.loss(output, target)
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data).data
                
                # bic의 end_KD = mid = start
                # bic의 start_KD = end_KD - stepsize
                
                
                if bft is False:
                    
                    score = score[:,:start].data
                
#                     soft_target = F.softmax(score / T, dim=1)
#                     output_log = F.log_softmax(output[:,:start] / T, dim=1)
#                     loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                    
                    loss_KD = torch.zeros(tasknum).cuda()
                    for t in range(tasknum):

                        # local distillation
                        start_KD = (t) * self.args.step_size
                        end_KD = (t+1) * self.args.step_size

                        soft_target = F.softmax(score[:,start_KD:end_KD] / T, dim=1)
                        output_log = F.log_softmax(output[:,start_KD:end_KD] / T, dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                    loss_KD = loss_KD.sum()
                else:
                    soft_target = F.softmax(score[:,start:end] / T, dim=1)
                    output_log = F.log_softmax(output[:,start:end] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                
                
            self.optimizer.zero_grad()
            (0.5*loss_KD + loss_CE).backward()
            self.optimizer.step()
           