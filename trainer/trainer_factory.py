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
    def get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer):
        
        if args.trainer == 'lwf':
            import trainer.lwf as trainer
        elif args.trainer == 'er':
            import trainer.er as trainer
        elif args.trainer == 'bayes':
            import trainer.bayes as trainer
        elif args.trainer == 'gda':
            import trainer.GDA as trainer
        elif args.trainer == 'coreset':
            import trainer.coreset as trainer
        elif args.trainer == 'icarl':
            import trainer.icarl as trainer
        elif args.trainer == 'ood':
            import trainer.ood as trainer
        elif args.trainer == 'bin_finetune':
            import trainer.bin_finetune as trainer
        return trainer.Trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)
    
class CutMixCollator:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        batch = self.cutmix(batch, self.alpha)
        return batch
    
    def cutmix(self,batch, alpha):
        data, targets = batch

#         indices = torch.randperm(data.size(0)).cuda()
        indices = np.random.permutation(data.size(0))
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
            img = self.loader(img)
            
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