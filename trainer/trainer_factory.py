import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(train_iterator, myModel, args, optimizer):
        
        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'ssil':
            import trainer.ssil as trainer
        elif args.trainer == 'ft' or args.trainer == 'il2m':
            import trainer.ft as trainer
        elif args.trainer == 'icarl':
            import trainer.icarl as trainer
        elif args.trainer == 'bic':
            import trainer.bic as trainer
        elif args.trainer == 'wa':
            import trainer.wa as trainer
        elif args.trainer == 'ft_bic':
            import trainer.ft_bic as trainer
        elif args.trainer == 'ft_wa':
            import trainer.ft_wa as trainer
        elif args.trainer == 'eeil':
            import trainer.eeil as trainer
        elif args.trainer == 'ft_bic_focal':
            import trainer.ft_bic_focal as trainer
        elif args.trainer == 'FCft':
            import trainer.FCft as trainer
        elif args.trainer == 'ft_lsm':
            import trainer.ft_lsm as trainer
        return trainer.Trainer(train_iterator, myModel, args, optimizer)
    

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
            img = self.loader(img)
            
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labelsNormal[index]

class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, trainDataIterator, model, args, optimizer):
        self.train_data_iterator = trainDataIterator
        self.model = model
        self.args = args
        self.incremental_loader = self.train_data_iterator.dataset
        self.optimizer = optimizer
        self.model_fixed = copy.deepcopy(self.model)
        for param in self.model_fixed.parameters():
            param.requires_grad = False
        self.current_lr = args.lr
        
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


