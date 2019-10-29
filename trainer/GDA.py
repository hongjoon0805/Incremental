# GDA 기반의 incremental learning 코드 짜기
# output linaer model에 bias가 있을 수 있으니 linear model로 classify 하지 말고 generative model로 classify 하자
# feature space regularization이 무조건 있어야한다
# Class incremental learning은 결국 linear classification이다. 그래서 Fisher's linear discriminant를 많이 참고하면 좋을 듯하다.
# Within class variance를 줄이고 between class variance를 키우는게 좋다.
# UCL은 weight space에 noise를 주었는데, Incremental learning에서는 feature space에 noise를 주어 중요한 dimension을 확보해야겠다. 
# Feature space에서 regularization을 줄 수 있는 Bayesian online learning framework를 사용하자.

from __future__ import print_function

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bayes_layer import NoiseLayer
import math

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

        print("Shuffling turned off for debugging")
        # random.seed(args.seed)
        # random.shuffle(self.all_classes)


class Trainer(GenericTrainer):
    def __init__(self, trainDataIterator, testDataIterator, dataset, model, args, optimizer):
        super().__init__(trainDataIterator, testDataIterator, dataset, model, args, optimizer)

    def update_lr(self, epoch):
        for temp in range(0, len(self.args.schedule)):
            if self.args.schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * self.args.gammas[temp]
                    print("Changing learning rate from %0.4f to %0.4f"%(self.current_lr,
                                                                        self.current_lr * self.args.gammas[temp]))
                    self.current_lr *= self.args.gammas[temp]

    def increment_classes(self):
        '''
        Add classes starting from class_group to class_group + step_size 
        :param class_group: 
        :return: N/A. Only has side-affects 
        '''

    def setup_training(self):
        
        self.train_data_iterator.dataset.update_exemplar()
        
        self.train_data_iterator.dataset.task_change()
        self.test_data_iterator.dataset.task_change()
        
        for param_group in self.optimizer.param_groups:
            print("Setting LR to %0.4f"%self.args.lr)
            param_group['lr'] = self.args.lr
            self.current_lr = self.args.lr

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
        classes = self.args.step_size * (tasknum+1)
        for data, y, target in tqdm(self.train_data_iterator):
            data, y, target = data.cuda(), y.cuda(), target.cuda()
            
            if tasknum > 0:
                data_r, y_r, target_r = self.train_data_iterator.dataset.sample_exemplar()
                data_r, y_r, target_r = data_r.cuda(), y_r.cuda(), target_r.cuda()
                
                data = torch.cat((data,data_r))
                y = torch.cat((y,y_r))
                target = torch.cat((target,target_r))
                
            y_onehot = torch.FloatTensor(len(target), self.dataset.classes).cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)
            target = target.squeeze()
            
            output, features = self.model(data, feature_return=True)
            score, saver_features = self.model_fixed(data, feature_return=True)
            
            start = 0
            end = (tasknum+1) * self.args.step_size
            
            # classification loss
            output_log = F.log_softmax(output[:,start:end], dim=1)
            loss_CE = F.kl_div(output_log, y_onehot[:,start:end], reduction='batchmean')
            
            # feature space regularization
            loss_features = self.feature_regularization(features, saver_features, tasknum)
            
            # compute class conditional mean & total mean & totalFeatures
            class_means, means, totalFeatures = self.compute_means(tasknum, features, target)
            
            # compute within class covariance
            loss_S_W = self.within_var(features, target, class_means)
            
            # compute between class covariance
            loss_S_B = self.between_var(class_means, means, totalFeatures)
            
            # SGD
            self.optimizer.zero_grad()
            (loss_CE + loss_features + loss_S_W/loss_S_B * self.args.alpha).backward()
            self.optimizer.step()
        
        print(loss_CE)
#         print(loss_S_W/loss_S_B * 0.1)

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))
        
    def feature_regularization(self, trainer_features, saver_features, tasknum):
        
        sigma_reg = 0
        feature_reg = 0
        
        saved = 0
        if tasknum>0:
            saved = 1
            
        for (_, saver_layer), (_, trainer_layer) in zip(self.model_fixed.named_children(), self.model.named_children()):
            if isinstance(trainer_layer, NoiseLayer)==False:
                continue
            
            in_features = saver_layer.rho.shape[0]
            
            std_init = math.sqrt(in_features * self.args.ratio)
            trainer_sigma = torch.log1p(torch.exp(trainer_layer.rho))
            saver_sigma = torch.log1p(torch.exp(saver_layer.rho))
            
            feature_reg_strength = std_init / saver_sigma
            
            featue_reg = (feature_reg_strength * (trainer_features - saver_features)).norm(2) ** 2
            
            grace_forget = (trainer_sigma / 100) ** 2 - torch.log((trainer_sigma / 100) ** 2).sum()
            sigma_reg = ((trainer_sigma / saver_sigma) ** 2 - torch.log((trainer_sigma / saver_sigma) ** 2)).sum()
            
            sigma_grace_forget_reg = sigma_reg + grace_forget
            
#         return self.args.beta * sigma_reg + saved * feature_reg
        return sigma_reg + saved * feature_reg
        
    def compute_means(self, tasknum, features, target):
        # feature normalize?
        
        classes = self.args.step_size * (tasknum+1)
        class_means = torch.zeros((classes, self.model.featureSize)).cuda()
        totalFeatures = torch.zeros((classes, 1)).cuda()

        class_means.index_add_(0,target,features)
        totalFeatures.index_add_(0,target,torch.ones(target.shape[0]))
        
        means = (class_means * totalFeatures).sum(dim=0) / target.shape[0]
        
        return class_means, means, totalFeatures
    
    def within_var(self, features, target, class_means):
        classes = class_means.shape[0]
        featureSize = features.shape[1]
        batchSize = features.shape[0]
        xn_mk = features-class_means[target]
        S_W_batch = torch.bmm(xn_mk.unsqueeze(2), xn_mk.unsqueeze(1))
        S_W_sum = torch.abs(S_W_batch).sum()
        return S_W_sum
    
    def between_var(self, class_means, means, totalFeatures):
        mk_m = class_means - means
        S_B_batch = torch.bmm(mk_m.unsqueeze(2), mk_m.unsqueeze(1)) * totalFeatures.unsqueeze(2)
        S_B_sum = torch.abs(S_B_batch).sum()
        return S_B_sum
        