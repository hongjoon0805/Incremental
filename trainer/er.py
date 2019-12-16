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
import torch.nn as nn
import torch.utils.data as td
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

        return img, self.labels[index], self.labelsNormal[index]

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
#         start = self.train_data_iterator.dataset.start
        start = 0
        end = self.train_data_iterator.dataset.end
        mid = end-self.args.step_size
        kwargs = {'num_workers': 32, 'pin_memory': True}
        exemplar_dataset_loaders = ExemplarLoader(self.train_data_iterator.dataset)
        exemplar_iterator = torch.utils.data.DataLoader(exemplar_dataset_loaders,
                                                        batch_size=self.args.replay_batch_size, 
                                                        shuffle=True, drop_last=True, **kwargs)
        
        
        if tasknum > 0:
            iterator = zip(self.train_data_iterator, exemplar_iterator)
        else:
            iterator = self.train_data_iterator
            
        for samples in tqdm(iterator):
            if tasknum > 0:
                curr, prev = samples
                
                data, y, target = curr
                data, y, target = data.cuda(), y.cuda(), target.cuda()
                
                batch_size = data.shape[0]
                
                data_r, y_r, target_r = prev
                data_r, y_r, target_r = data_r.cuda(), y_r.cuda(), target_r.cuda()
                
                replay_size = data_r.shape[0]
                
                data = torch.cat((data,data_r))
                y = torch.cat((y,y_r))
                target = torch.cat((target,target_r))
                
            else:
                data, y, target = samples
                data, y, target = data.cuda(), y.cuda(), target.cuda()
            
                batch_size = data.shape[0]
            
            y_onehot = torch.FloatTensor(len(target), self.dataset.classes).cuda()

            y_onehot.zero_()
            target.unsqueeze_(1)
            y_onehot.scatter_(1, target, 1)
            
            uniform = torch.ones_like(y_onehot)
            
            output, bin_out = self.model(data, bc=True)
            
            if self.args.prev_new:
                loss_CE_curr = 0
                loss_CE_prev = 0

                curr = output[:batch_size,mid:end]
                curr_log = F.log_softmax(curr, dim=1)
                loss_CE_curr = F.kl_div(curr_log, y_onehot[:batch_size,mid:end], reduction='sum')


                if tasknum > 0:
                    prev = output[batch_size:batch_size+replay_size,start:mid]
                    prev_log = F.log_softmax(prev, dim=1)
                    loss_CE_prev = F.kl_div(prev_log, y_onehot[batch_size:batch_size+replay_size,start:mid], reduction='sum')

                    loss_CE = (loss_CE_curr + loss_CE_prev*self.args.alpha) / (batch_size + replay_size)
                    
                    if self.args.uniform_penalty:
#                         curr_uni = output[:batch_size,start:mid]
#                         curr_uni_log = F.log_softmax(curr_uni, dim=1)
#                         loss_uni_curr = F.kl_div(curr_uni_log, uniform[:batch_size,start:mid] / (mid-start),
#                                                  reduction='sum')
                        
                        prev_uni = output[batch_size:batch_size+replay_size,mid:end]
                        prev_uni_log = F.log_softmax(prev_uni, dim=1)
                        loss_uni_prev = F.kl_div(prev_uni_log, uniform[batch_size:batch_size+replay_size,mid:end] / (end-mid),
                                                 reduction='sum')
#                         loss_CE = loss_CE + (loss_uni_curr + loss_uni_prev) / (batch_size + replay_size)
                        loss_CE = loss_CE + (loss_uni_prev) / (replay_size)
                    
                else:
                    loss_CE = loss_CE_curr / batch_size
            else:
                output_log = F.log_softmax(output[:,:end], dim=1)
                loss_CE = F.kl_div(output_log, y_onehot[:,:end])

            # prev: 1
            # new : 0
            loss_BCE = 0
            if tasknum > 0:
                binary_target = torch.zeros(batch_size + replay_size).cuda()
                binary_target[batch_size:batch_size+replay_size] = 1
                # Binary Classification using sigmoid output
                if self.args.bin_sigmoid:
                    prev_prob = F.sigmoid(bin_out).squeeze()
                    loss_BCE = F.binary_cross_entropy(prev_prob, binary_target)

                # Binary Classification using softmax output
                elif self.args.bin_softmax:
                    prev_prob = F.softmax(output[:,:end], dim=1)[:,:mid].sum(dim=1)
                    loss_BCE = F.binary_crosse_entropy(prev_prob, binary_target)
            
            loss_KD = 0
            # 일단 local distillation은 보류.

            self.optimizer.zero_grad()
            (loss_KD + loss_CE + loss_BCE).backward()
            self.optimizer.step()

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))

            ###############################################################################################################
            # gradient scale
            # gradient의 norm을 출력 해봐야한다. 진짜로 norm의 차이가 큰지 확인해봐야함.
#             if tasknum>0:
#                 y_onehot[self.args.batch_size:] *= self.args.alpha
            ###############################################################################################################

    
#             if tasknum > 0:
#                 data_r, y_r, target_r = self.train_data_iterator.dataset.sample_exemplar()
#                 data_r, y_r, target_r = data_r.cuda(), y_r.cuda(), target_r.cuda()
                
#                 data = torch.cat((data,data_r))
#                 y = torch.cat((y,y_r))
#                 target = torch.cat((target,target_r))
            # prev: 0~end
            # curr: mid~end
            # 이렇게 하면 prev가 curr로 classify 되는 경우가 사라지지 않을까?
            
#             loss_CE_curr = 0
#             loss_CE_prev = 0
            
#             curr = output[:size,mid:end]
#             curr_log = F.log_softmax(curr, dim=1)
#             loss_CE_curr = F.kl_div(curr_log, y_onehot[:size,mid:end], reduction='sum')
            
            
#             if tasknum > 0:
#                 prev = output[size:size*2,start:mid]
#                 prev_log = F.log_softmax(prev, dim=1)
#                 loss_CE_prev = F.kl_div(prev_log, y_onehot[size:size*2,start:mid], reduction='sum')
                
#                 loss_CE = (loss_CE_curr + loss_CE_prev) / (2*size)
#             else:
#                 loss_CE = loss_CE_curr / size


#             if tasknum > 0:
#                 score = self.model_fixed(data).data
#                 loss_KD = torch.zeros(tasknum).cuda()
#                 for t in range(tasknum):
                    
#                     # local distillation
#                     start = (t) * self.args.step_size
#                     end = (t+1) * self.args.step_size

#                     soft_target = F.softmax(score[:,start:end] / T, dim=1)
#                     output_log = F.log_softmax(output[:,start:end] / T, dim=1)
#                     loss_KD[t] = F.kl_div(output_log, soft_target) * (T**2) * self.args.alpha
                
#                 loss_KD = loss_KD.sum()