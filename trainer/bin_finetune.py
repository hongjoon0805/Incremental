''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from __future__ import print_function

import copy
import logging

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
import torchvision.transforms.functional as trnF
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

import networks

class ExemplarLoader(td.Dataset):
    def __init__(self, train_dataset):
        
        self.data = train_dataset.data
        self.labels = train_dataset.labels
        self.labelsNormal = train_dataset.labelsNormal
        self.exemplar = train_dataset.exemplar
        self.transform = train_dataset.transform
        
        self.loader = train_dataset.loader
        self.mem_sz = len(self.exemplar)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        index = self.exemplar[index % self.mem_sz]
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = self.loader(img[0])
        
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]
    

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
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
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
        tasknum = self.train_data_iterator.dataset.t
        model_name = 'models/trained_model/RESULT_{}_{}_{}_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_128_epoch_100_factor_5_RingBuffer_CE_lr_change_task_{}.pt'.format(self.args.dataset, self.args.option, self.args.seed, self.args.memory_budget, self.args.base_classes, tasknum)
        
        self.model.load_state_dict(torch.load(model_name))
#         self.model.module.binary_fc = nn.Linear(512, 2)
        self.model.module.binary_fc = nn.Linear(512, 1)
        
        self.optimizer = torch.optim.SGD(self.model.module.binary_fc.parameters(), self.args.lr, momentum=self.args.momentum,
                            weight_decay=self.args.decay, nesterov=True)

    def get_model(self):
        myModel = networks.ModelFactory.get_model(self.args.dataset).cuda()
        optimizer = torch.optim.SGD(myModel.parameters(), self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.decay, nesterov=True)
        myModel.eval()

        self.current_lr = self.args.lr

        self.model_single = myModel
        self.optimizer_single = optimizer

    def train(self, epoch):
        
        self.model.train()
        print("Epochs %d"%epoch)
        
        tasknum = self.train_data_iterator.dataset.t
        kwargs = {'num_workers': 32, 'pin_memory': True}
        exemplar_dataset_loaders = ExemplarLoader(self.train_data_iterator.dataset)
        exemplar_iterator = torch.utils.data.DataLoader(exemplar_dataset_loaders,
                                                        batch_size=self.args.replay_batch_size, 
                                                        shuffle=True, drop_last=True, **kwargs)
        
        iterator = zip(self.train_data_iterator, exemplar_iterator)
        for samples in tqdm(iterator):
            curr, prev = samples

            data, _ = curr
            data = data.cuda()
            batch_size = data.shape[0]

            data_r, _ = prev
            data_r = data_r.cuda()

            replay_size = data_r.shape[0]

            data = torch.cat((data,data_r))
                
            _, bin_out = self.model(data, bc=True)
            binary_target = torch.zeros(batch_size + replay_size).cuda()
            binary_target[:batch_size] = 1
            
            loss_BCE = F.binary_cross_entropy_with_logits(bin_out.squeeze(), binary_target)
            
            self.optimizer.zero_grad()
            (loss_BCE).backward()
            self.optimizer.step()
