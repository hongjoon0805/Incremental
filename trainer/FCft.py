from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import networks
import trainer
from torch.nn import init

class Trainer(trainer.GenericTrainer):
    def __init__(self, trainDataIterator, model, args, optimizer):
        super().__init__(trainDataIterator, model, args, optimizer)
        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.fc_optimizer = torch.optim.SGD(self.model.parameters(), args.lr, weight_decay=args.decay)
#         self.fc_optimizer = torch.optim.SGD(self.model.parameters(), args.lr)

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
            
        for param in self.model.parameters():
            param.requires_grad = True
    
    def setup_fc_training(self):
        
        # rand-init
        init.kaiming_normal(self.model.module.fc.weight)
        self.model.module.fc.bias.data.zero_()
        
        for param_group in self.fc_optimizer.param_groups:
            print("Setting LR to %0.4f"%0.001)
            param_group['lr'] = 0.001
            self.current_lr = 0.001
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.module.fc.parameters():
            param.requires_grad = True

    def train(self, epoch, FC_retrain = 0):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        T=2
        tasknum = self.train_data_iterator.dataset.t
        end = self.train_data_iterator.dataset.end
        mid = end - self.args.step_size
        start = 0
        
        lamb = mid / end
        total = 0
        loss_sum = 0
        acc = 0
        
        if FC_retrain == 0:
            optimizer = self.optimizer
        elif FC_retrain == 1:
            optimizer = self.fc_optimizer
        
        for data, target in tqdm(self.train_data_iterator):
            data, target = data.cuda(), target.cuda()
            
            output = self.model(data)
            batch_size = data.shape[0]
            total += batch_size
            
            
            loss_CE = self.loss(output[:,:end], target)
            if FC_retrain == 1:
                # loss
                loss_sum += loss_CE.item() * batch_size
                
                # accuracy
                pred_1 = output.data.max(1, keepdim=True)[1]
                correct_1 = pred_1.eq(target.data.view_as(pred_1)).sum().item()
                acc += correct_1
            loss_KD = 0
            if tasknum > 0 and FC_retrain == 0:
                end_KD = mid
                score = self.model_fixed(data)[:,:end_KD]

                soft_target = F.softmax(score / T, dim=1)
                output_log = F.log_softmax(output[:,:end_KD] / T, dim=1)
                loss_KD = F.kl_div(output_log, soft_target, reduction='batchmean') * T * T
            
            optimizer.zero_grad()
            (lamb*loss_KD + (1-lamb)*loss_CE).backward()
            optimizer.step()
        
        if FC_retrain == 1:
            print('Loss: ', loss_sum / total)
            print('Accuracy: ', acc / total)
