
from __future__ import print_function

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

import networks

import sys
sys.path.append('..')
from bayes_layer import BayesianLinear, BayesianConv2D, _calculate_fan_in_and_fan_out

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
         
        self.model_fixed = networks.ModelFactory.get_model(self.args.dataset, self.args.ratio).cuda()
        state = model.state_dict()
        state_clone = copy.deepcopy(state)
        self.model_fixed.load_state_dict(state_clone)
        
        self.active_classes = []
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models = []
        self.current_lr = args.lr
        self.all_classes = list(range(dataset.classes))
        self.all_classes.sort(reverse=True)
        
        self.model_single = networks.ModelFactory.get_model(self.args.dataset, self.args.ratio).cuda()
        state = model.state_dict()
        state_clone = copy.deepcopy(state)
        self.model_single.load_state_dict(state_clone)
        
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

    def setup_training(self):
        
#         self.train_data_iterator.dataset.update_exemplar()
        
        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.2f"%self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr

    def update_frozen_model(self):
        self.model.eval()
        
        self.model_fixed = networks.ModelFactory.get_model(self.args.dataset, self.args.ratio).cuda()
        state = self.model.state_dict()
        state_clone = copy.deepcopy(state)
        self.model_fixed.load_state_dict(state_clone)
        
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.models.append(self.model_fixed)

    def get_model(self):
        myModel = networks.ModelFactory.get_model(self.args.dataset, self.args.ratio).cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        
#         if self.args.trainer == 'bayes':
#             optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def train(self, epoch):
        
        T=2
        
        self.model.train()
        
        start = self.train_data_iterator.dataset.start
        end = self.train_data_iterator.dataset.end
        
        print("Epochs %d"%epoch)
        tasknum = self.train_data_iterator.dataset.t
        for data, y, target in tqdm(self.train_data_iterator):
            data, y, target = data.cuda(), y.cuda(), target.cuda()
            
            y_onehot = torch.FloatTensor(len(target), self.dataset.classes).cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)
            
            output = self.model(data, sample=True)
            
            start = tasknum * self.args.step_size
            end = (tasknum+1) * self.args.step_size
            
            output_log = F.log_softmax(output[:,start:end], dim=1)
            loss_CE = F.kl_div(output_log, y_onehot[:,start:end])
            
            loss_KD = 0
            if tasknum > 0:
                loss_KD = torch.zeros((self.args.sample,tasknum)).cuda()
                for s in range(self.args.sample):
                    score = self.model_fixed(data, sample=True)
                    for t in range(tasknum):
                        
                        # local distillation
                        start = (t) * self.args.step_size
                        end = (t+1) * self.args.step_size
                        
                        soft_target = F.softmax(score[:,start:end] / T, dim=1)
                        output_log = F.log_softmax(output[:,start:end] / T, dim=1)
                        
                        loss_KD[s][t] = F.kl_div(output_log, soft_target) * (T**2)
                        
                    loss_KD[s] = loss_KD[s].sum()
                    
                    # global distillation
#                     start = 0
#                     end = (tasknum) * self.args.step_size

#                     soft_target = F.softmax(score[:,start:end] / T, dim=1)
#                     output_log = F.log_softmax(output[:,start:end] / T, dim=1)
#                     loss_KD[s] = loss_KD[s] + F.kl_div(output_log, soft_target) * (T**2)
                    
                    
                loss_KD = loss_KD.mean()
            
            reg_loss = self.custom_regularization(self.model_fixed, self.model, tasknum).cuda()
            
            self.optimizer.zero_grad()
            (loss_KD + loss_CE + reg_loss).backward()
            self.optimizer.step()

    def add_model(self):
        
        model = networks.ModelFactory.get_model(self.args.dataset, self.args.ratio).cuda()
        state = self.model_single.state_dict()
        state_clone = copy.deepcopy(state)
        model.load_state_dict(state_clone)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))
        
    def custom_regularization(self, saver_net, trainer_net, tasknum):
        
        sigma_weight_reg_sum = 0
        sigma_bias_reg_sum = 0
        sigma_weight_normal_reg_sum = 0
        sigma_bias_normal_reg_sum = 0
        mu_weight_reg_sum = 0
        mu_bias_reg_sum = 0
        L1_mu_weight_reg_sum = 0
        L1_mu_bias_reg_sum = 0
        loss = 0
        
        saved=0
        if tasknum>0:
            saved=1
        
        else:
            prev_weight_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(0,0))
        
        for (_, saver_layer), (_, trainer_layer) in zip(saver_net.named_children(), trainer_net.named_children()):
            if isinstance(trainer_layer, BayesianLinear)==False and isinstance(trainer_layer, BayesianConv2D)==False:
                continue
            # calculate mu regularization
            trainer_weight_mu = trainer_layer.weight_mu
            saver_weight_mu = saver_layer.weight_mu
            trainer_bias = trainer_layer.bias
            saver_bias = saver_layer.bias
            
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_weight_mu)
            
            trainer_weight_sigma = torch.log1p(torch.exp(trainer_layer.weight_rho))
            saver_weight_sigma = torch.log1p(torch.exp(saver_layer.weight_rho))
            
            
            if isinstance(trainer_layer, BayesianLinear):
                std_init = math.sqrt((2 / fan_in) * self.args.ratio)
            if isinstance(trainer_layer, BayesianConv2D):
                std_init = math.sqrt((2 / fan_out) * self.args.ratio)
            
            saver_weight_strength = std_init / saver_weight_sigma
            
            L2_strength = saver_weight_strength
            bias_strength = torch.squeeze(saver_weight_strength)
            
            L1_sigma = saver_weight_sigma
            bias_sigma = torch.squeeze(saver_weight_sigma)
            
            
            mu_weight_reg = (L2_strength * (trainer_weight_mu-saver_weight_mu)).norm(2)**2
            mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2
            
            L1_mu_weight_reg = (torch.div(saver_weight_mu**2,L1_sigma**2)*(trainer_weight_mu - saver_weight_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)
            
            L1_mu_weight_reg = L1_mu_weight_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
            
            weight_sigma = (trainer_weight_sigma**2 / saver_weight_sigma**2)
            
            normal_weight_sigma = trainer_weight_sigma**2
            
            sigma_weight_reg_sum = sigma_weight_reg_sum + (weight_sigma - torch.log(weight_sigma)).sum()
            sigma_weight_normal_reg_sum = sigma_weight_normal_reg_sum + (normal_weight_sigma - torch.log(normal_weight_sigma)).sum()
            
            mu_weight_reg_sum = mu_weight_reg_sum + mu_weight_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            L1_mu_weight_reg_sum = L1_mu_weight_reg_sum + L1_mu_weight_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg
            
        
        # L2 loss
#         loss = loss + self.args.decay * (mu_weight_reg_sum + mu_bias_reg_sum)
        # L1 loss
#         loss = loss + saved * (L1_mu_weight_reg_sum + L1_mu_bias_reg_sum)
        # sigma regularization
        loss = loss + self.args.beta * (sigma_weight_reg_sum + sigma_weight_normal_reg_sum)
            
        return loss



        
        

