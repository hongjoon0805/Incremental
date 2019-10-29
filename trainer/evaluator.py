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
from numpy.linalg import inv

logger = logging.getLogger('iCARL')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="nmc", classes=100):
        if testType == "trainedClassifier":
            return softmax_evaluator()
        if testType == "generativeClassifier":
            return GDA(classes)

class GDA():

    def __init__(self, classes):
        self.classes = classes
    
    def update_moment(self, model, train_loader, step_size):
        
        # Set the mean to zero
        tasknum = train_loader.dataset.t
        
        # compute means
        classes = step_size * (tasknum+1)
        class_means = np.zeros((classes, model.featureSize), dtype=np.float32)
        totalFeatures = np.zeros((classes, 1), dtype=np.float32)
        
        # Iterate over all train Dataset
        for data, y, target in train_loader:
            data = data.cuda()
            _, features = model.forward(data, feature_return=True)
            featuresNp = features.data.cpu().numpy()
            
            if tasknum > 0:
                data_r, y_r, target_r = train_loader.dataset.sample_exemplar()
                data_r = data_r.cuda()

                _,_,data_r_feature = model.forward(data_r, sample=True)

                features = torch.cat((features, data_r_features))
                target = np.hstack((target,target_r))
            
            np.add.at(class_means, target, featuresNp)
            np.add.at(totalFeatures, target, 1)

        
        class_means = class_means / totalFeatures
        
        # compute precision
        covariance = np.zeros((model.featureSize, model.featureSize),dtype=np.float32)
        
        for data, y, target in train_loader:
            data = data.cuda()
            _, features = model.forward(data, feature_return=True)
            featuresNp = features.data.cpu().numpy()
            
            if tasknum > 0:
                
                data_r, y_r, target_r = train_loader.dataset.sample_exemplar()
                data_r = data_r.cuda()

                _,_,data_r_feature = model.forward(data_r, sample=True)

                features = torch.cat((features, data_r_features))
                target = np.hstack((target,target_r))
            
            vec = featuresNp - class_means[target]
            np.expand_dims(vec, axis=2)
            cov = np.matmul(np.expand_dims(vec, axis=2), np.expand_dims(vec, axis=1)).sum(axis=0)
            covariance += cov
        
        covariance = covariance / totalFeatures.sum()
        print(covariance)
        precision = inv(covariance)
        
        self.class_means = torch.from_numpy(class_means).cuda()
        self.precision = torch.from_numpy(precision).cuda()
        
        return
    
    def evaluate(self, model, loader, tasknum, step_size, mode='train'):
        
        model.eval()

        correct = 0
        
        for data, y, target in loader:
            data, target = data.cuda(), target.cuda()
            _, features  = model(data, feature_return=True)
            
            # M_distance: NxC(start~end)
            #wwww features: NxD
            # features - mean: NxC(start~end)xD
            start = 0
            if mode == 'train':
                start = tasknum * step_size
                target = target%step_size
            end = (tasknum+1) * step_size
            
            batch_vec = (features.unsqueeze(1) - self.class_means[start:end].unsqueeze(0)).view(-1,features.shape[1])
            temp = torch.matmul(batch_vec,self.precision).unsqueeze(1)
            Mahalanobis = torch.matmul(temp,batch_vec.unsqueeze(2)).view(-1,end-start)
            print(Mahalanobis.shape)
            _, pred = torch.min(Mahalanobis,1)
            print(pred.shape)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            self.class_means = self.class_means.squeeze()

        return 100. * correct / len(loader.dataset)

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
