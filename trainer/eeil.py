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
        self.bft_optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum)
        

    def update_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]
    
    def update_bft_lr(self, epoch, schedule):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.bft_optimizer.param_groups:
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
        # ?
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr
            
        lr = lr/10
        self.bft_optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr/10, momentum=self.args.momentum)
        """ # why this code?
        for param_group in self.bft_optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr
        """
    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
    
    def balance_fine_tune(self):
        self.train_data_iterator.dataset.update_bft_buffer()
        self.train_data_iterator.dataset.mode = 'b-ft'
        # balanced fine tuning - exemplar가 new old 개수 동일하게 되어있는상태
        schedule = np.array(self.args.schedule)
        #bftepoch = self.args.bftepoch
        bftepoch = int(self.args.nepochs*0.75)
        for epoch in range(bftepoch):
            self.update_bft_lr(epoch, schedule)
            self.train(epoch, bft=True)
        
        self.train_data_iterator.dataset.mode = 'train'
    
    def train(self, epoch, bft=False):
        
        T=2
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        
        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            loss_CE = self.loss(output, target)
            
            loss_KD = 0
            if tasknum > 0:
                score = self.model_fixed(data).data
                
                if bft is False:
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
                    score = self.model(data).data
                    soft_target = F.softmax(score[:,start:end] / T, dim=1)
                    output_log = F.log_softmax(output[:,start:end] / T, dim=1)
                    loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                
                
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()
           