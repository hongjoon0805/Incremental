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
        self.alpha =  nn.Parameter(torch.Tensor(1).uniform_(1,1))
        self.beta = nn.Parameter(torch.Tensor(1).uniform_(0,0))
    def forward(self, x):
        return x*self.alpha + self.beta

class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.bias_correction_layer = BiasLayer()
        
        self.bias_optimizer = torch.optim.Adam(self.bias_correction_layer.parameters(), 0.001)

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
                for param_group in self.bias_optimizer.param_groups:
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
        
        self.bias_correction_layer = BiasLayer()
        self.bias_optimizer = torch.optim.Adam(self.bias_correction_layer.parameters(), 0.001)
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%lr)
            param_group['lr'] = lr
            self.current_lr = lr
        
        lr = lr/100
        for param_group in self.bias_optimizer.param_groups:
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
#                 loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean')
                
            self.optimizer.zero_grad()
            (lamb*loss_KD + (1-lamb)*loss_CE).backward()
            self.optimizer.step()
            
    def train_bias_correction(self, bias_iterator):
        
        self.model.eval()
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        start = end-self.args.step_size
        
        print('start, end: ',start,end)
        
        loss = 0
        
        prev_prev_sum = 0
        prev_new_sum = 0
        
        new_prev_sum = 0
        new_new_sum = 0
        
        prev_cnt = 0
        new_cnt = 0
        
        correct_before = 0
        correct_after = 0
        
        correct_prev_before = 0
        correct_prev_after = 0
        correct_new_before = 0
        correct_new_after = 0
        
        for data, target in tqdm(bias_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)[:,:end]
            with torch.no_grad():
                prev_idx = target < start
                new_idx = target >= start
                prev_cnt += prev_idx.float().sum().item()
                new_cnt += new_idx.float().sum().item()

                prev_prev_sum += output[prev_idx][:,:start].sum().item()
                prev_new_sum += output[prev_idx][:,start:].sum().item()
                new_prev_sum += output[new_idx][:,:start].sum().item()
                new_new_sum += output[new_idx][:,start:].sum().item()
                
                if prev_idx.float().sum() > 0:
                    pred_prev = output[prev_idx].data.max(1, keepdim=True)[1]
                    correct_prev_before += pred_prev.eq(target[prev_idx].data.view_as(pred_prev)).sum().item()
                
                if new_idx.float().sum() > 0:
                    pred_new = output[new_idx].data.max(1, keepdim=True)[1]
                    correct_new_before += pred_new.eq(target[new_idx].data.view_as(pred_new)).sum().item()
                
                pred = output.data.max(1, keepdim=True)[1]
                correct_before += pred.eq(target.data.view_as(pred)).sum().item()
                
            output_new = self.bias_correction_layer(output[:,start:end])
            output = torch.cat((output[:,:start], output_new), dim=1)
            
            if prev_idx.float().sum() > 0:
                pred_prev = output[prev_idx].data.max(1, keepdim=True)[1]
                correct_prev_after += pred_prev.eq(target[prev_idx].data.view_as(pred_prev)).sum().item()
            if new_idx.float().sum() > 0:
                pred_new = output[new_idx].data.max(1, keepdim=True)[1]
                correct_new_after += pred_new.eq(target[new_idx].data.view_as(pred_new)).sum().item()
            
            pred = output.data.max(1, keepdim=True)[1]
            correct_after += pred.eq(target.data.view_as(pred)).sum().item()
            
            loss_CE = self.loss(output, target)
            
            self.bias_optimizer.zero_grad()
            (loss_CE).backward()
            self.bias_optimizer.step()
            
            loss += loss_CE.item()
        print('Loss: ',loss/4)
        print('Acc before: ', correct_before / (prev_cnt + new_cnt), 'Acc after: ', correct_after / (prev_cnt + new_cnt))
        print('Acc prev before: ', correct_prev_before / (prev_cnt), 'Acc prev after: ', correct_prev_after / (prev_cnt))
        print('Acc new before: ', correct_new_before / (new_cnt), 'Acc new after: ', correct_new_after / (new_cnt))
        print('Prev Prev sum: ', prev_prev_sum/prev_cnt)
        print('Prev New sum: ', prev_new_sum/prev_cnt)
        print('New Prev sum: ', new_prev_sum/new_cnt)
        print('New New sum: ', new_new_sum/new_cnt)
        
#         print('Prev cnt: ', prev_cnt)
#         print('New cnt: ', new_cnt)
        