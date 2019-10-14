''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import confusionmeter

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="nmc"):
        if testType == "trainedClassifier":
            return softmax_evaluator()


class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self):
        pass

    def evaluate(self, model, loader, tasknum, step_size, mode = 'train'):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :return: 
        '''
        model.eval()
        correct = 0
        tempCounter = 0
        for data, y, target in loader:
            data, y, target = data.cuda(), y.cuda(), target.cuda()
            
            start = 0
            if mode == 'train':
                start = tasknum * step_size
                target = target%step_size
            end = (tasknum+1) * step_size
            
            output = model(data)[:,start:end]
            
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        return 100. * correct / len(loader.dataset)
