import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as td
from PIL import Image

class TrainerFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(train_iterator, myModel, args):
        
        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'ssil':
            import trainer.ssil as trainer
        elif args.trainer == 'ft' or args.trainer == 'il2m' or args.trainer == 'ft_nem':
            import trainer.ft as trainer
        elif args.trainer == 'icarl':
            import trainer.icarl as trainer
        elif args.trainer == 'full':
            import trainer.full as trainer
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
        elif args.trainer == 'vanilla':
            import trainer.vanilla as trainer
        elif args.trainer == 'ft_bic_focal':
            import trainer.ft_bic_focal as trainer
        elif args.trainer == 'FCft':
            import trainer.FCft as trainer
        elif args.trainer == 'gda':
            import trainer.gda as trainer
        elif args.trainer == 'ft_lsm':
            import trainer.ft_lsm as trainer
        return trainer.Trainer(train_iterator, myModel, args)
    

class ExemplarLoader(td.Dataset):
    def __init__(self, train_dataset):
        
        self.data = train_dataset.dataset.train_data
        self.labels = train_dataset.dataset.train_labels
        self.exemplar = train_dataset.exemplar
        self.transform = train_dataset.dataset.train_transform
        self.mem_sz = len(self.exemplar)

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
        index = self.exemplar[index % self.mem_sz]
        img = self.data[index]
        try:
            img = Image.fromarray(img)
        except:
            img = Image.open(img, mode='r').convert('RGB')
            
        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self, IncrementalLoader, model, args):
        self.incremental_loader = IncrementalLoader
        self.model = model
        self.args = args
        self.kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.incremental_loader.mode = 'train'
        self.train_iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, drop_last=True, 
                                                          shuffle=True, **self.kwargs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr, momentum=args.momentum,
                                         weight_decay=args.decay, nesterov=False)
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
        
        self.incremental_loader.update_exemplar()
        self.incremental_loader.task_change()

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

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, mode = 'CrossEntropy'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.mode = mode

    def forward(self, x, target):
        
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
#         output = x - index * self.m_list
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        if self.mode == 'CrossEntropy':
            return F.cross_entropy(self.s*output, target, weight=self.weight)
        
        elif self.mode == 'Hinge':
            # output 조작
            aux = x - index_float * np.inf
            max_idx = aux.max(1)[1]
            logit_index = torch.zeros_like(x, dtype=torch.uint8)
            logit_index.scatter_(1, max_idx.data.view(-1,1), 1)
            
            logit_index_float = logit_index.type(torch.cuda.FloatTensor)
            
            all_index = index_float + logit_index_float
            
            output = all_index * output - (1-all_index) * np.inf
            
            return F.multi_margin_loss(self.s*output, target, p=1, margin=0, weight=self.weight)
