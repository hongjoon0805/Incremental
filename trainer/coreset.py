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
from tqdm import tqdm

import networks

class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        self.train_data_iterator = trainDataIterator
        self.test_data_iterator = testDataIterator
        self.model = model
        self.args = args
        self.dataset = dataset
        self.train_loader = self.train_data_iterator.dataset
        self.older_classes = []
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models = []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        self.ce=torch.nn.CrossEntropyLoss()
        self.model_single = copy.deepcopy(self.model)
        self.optimizer_single = None


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)

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
        self.models.append(self.model_fixed)

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
        start = 0
        end = self.train_data_iterator.dataset.end
        mid = end-self.args.step_size
        
        for data, y, target in tqdm(self.train_data_iterator):
            data, y, target = data.cuda(), y.cuda(), target.cuda()
            
            oldClassesIndices = (target < (end -self.args.step_size)).int()
            old_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices > 0)).long())
            new_classes_indices = torch.squeeze(torch.nonzero((oldClassesIndices == 0)).long())
            
            y_onehot = torch.FloatTensor(len(target), self.dataset.classes).cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)
            
            uniform = torch.ones_like(y_onehot)
            
            output = self.model(data)
            
            if self.args.prev_new:
                loss_CE_curr = 0
                loss_CE_prev = 0

                curr = output[new_classes_indices,mid:end]
                curr_log = F.log_softmax(curr, dim=1)
                loss_CE_curr = F.kl_div(curr_log, y_onehot[new_classes_indices,mid:end], reduction='sum')


                if tasknum > 0:
                    prev = output[old_classes_indices,start:mid]
                    prev_log = F.log_softmax(prev, dim=1)
                    loss_CE_prev = F.kl_div(prev_log, y_onehot[old_classes_indices,start:mid], reduction='sum')
                    
                    loss_CE = (loss_CE_curr + loss_CE_prev*self.args.alpha) / (data.shape[0])
                    
                    if self.args.uniform_penalty:
                        curr_uni = output[new_classes_indices,start:mid]
                        curr_uni_log = F.log_softmax(curr_uni, dim=1)
                        loss_uni_curr = F.kl_div(curr_uni_log, uniform[new_classes_indices,start:mid] / (mid-start),
                                                 reduction='sum')
                        
                        prev_uni = output[old_classes_indices,mid:end]
                        prev_uni_log = F.log_softmax(prev_uni, dim=1)
                        loss_uni_prev = F.kl_div(prev_uni_log, uniform[old_classes_indices,mid:end] / (end-mid),
                                                 reduction='sum')
                        loss_CE = loss_CE + (loss_uni_curr + loss_uni_prev) / (data.shape[0])
                        
                else:
                    loss_CE = loss_CE_curr / data.shape[0]
            else:
                output_log = F.log_softmax(output[:,:end], dim=1)
                loss_CE = F.kl_div(output_log, y_onehot[:,:end])
            
            # 일단 local distillation은 보류.
            loss_KD = 0
            
            self.optimizer.zero_grad()
            (loss_KD + loss_CE).backward()
            self.optimizer.step()

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))
