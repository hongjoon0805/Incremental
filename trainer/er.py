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


class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = self.cutmix(batch, self.alpha)
        return batch
    
    def cutmix(self,batch, alpha):
        data, targets = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
        targets = (targets, shuffled_targets, lam)

    return data, targets


class CutMixCriterion:
    def __init__(self, reduction):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)

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
        self.ce=torch.nn.CrossEntropyLoss(reduction='sum')
        self.CutMux_CE = CutMixCriterion('sum')
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
                
                data, target = curr
                data, target = data.cuda(), target.cuda()
                
                batch_size = data.shape[0]
                
                data_r, target_r = prev
                data_r, target_r = data_r.cuda(), target_r.cuda()
                
                replay_size = data_r.shape[0]
                
                data = torch.cat((data,data_r))
                target = torch.cat((target,target_r))
                
            else:
                data, target = samples
                data, target = data.cuda(), target.cuda()
            
                batch_size = data.shape[0]
            
            y_onehot = torch.FloatTensor(len(target), self.dataset.classes).cuda()

            y_onehot.zero_()
            y_onehot.scatter_(1, target.unsqueeze(1), 1)
            
            output, bin_out = self.model(data, bc=True)
            
            loss_CE_curr = 0
            loss_CE_prev = 0

            curr = output[:batch_size,mid:end]
            loss_CE_curr = self.ce(curr, target[:batch_size]%(end-mid))
            
#             curr_log = F.log_softmax(curr, dim=1)
#             loss_CE_curr = F.kl_div(curr_log, y_onehot[:batch_size,mid:end], reduction='sum')

            if tasknum > 0:
                prev = output[batch_size:batch_size+replay_size,start:mid]
                loss_CE_prev = self.ce(prev, target[batch_size:batch_size+replay_size]%(mid-start))
#                 prev_log = F.log_softmax(prev, dim=1)
#                 loss_CE_prev = F.kl_div(prev_log, y_onehot[batch_size:batch_size+replay_size,start:mid], reduction='sum')
                loss_CE = (loss_CE_curr + loss_CE_prev) / (batch_size + replay_size)

            else:
                loss_CE = loss_CE_curr / batch_size

            # 일단 local distillation은 보류.

            self.optimizer.zero_grad()
            (loss_CE).backward()
            self.optimizer.step()

    def add_model(self):
        model = copy.deepcopy(self.model_single)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.models.append(model)
        print("Total Models %d"%len(self.models))
        
        
#             # prev: 1
#             # new : 0
#             loss_BCE = 0
#             if tasknum > 0:
#                 binary_target = torch.zeros(batch_size + replay_size).cuda()
#                 binary_target[batch_size:batch_size+replay_size] = 1
#                 # Binary Classification using sigmoid output
#                 if self.args.bin_sigmoid:
#                     prev_prob = F.sigmoid(bin_out).squeeze()
#                     loss_BCE = F.binary_cross_entropy(prev_prob, binary_target)

#                 # Binary Classification using softmax output
#                 elif self.args.bin_softmax:
#                     prev_prob = F.softmax(output[:,:end], dim=1)[:,:mid].sum(dim=1)
#                     loss_BCE = F.binary_crosse_entropy(prev_prob, binary_target)
