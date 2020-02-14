''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import logging

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from sklearn.metrics import roc_auc_score

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
        if testType == "binaryClassifier":
            return sigmoid_evaluator()
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
            data, target = data.cuda(), target.cuda()
            _, features = model.forward(data, feature_return=True)
            
            if tasknum > 0:
                data_r, y_r, target_r = train_loader.dataset.sample_exemplar()
                data_r, target_r = data_r.cuda(), target_r.cuda()

                _,_,data_r_feature = model.forward(data_r, sample=True)

                features = torch.cat((features, data_r_feature))
                target = torch.cat((target,target_r))
            
            featuresNp = features.data.cpu().numpy()
            np.add.at(class_means, target.data.cpu().numpy(), featuresNp)
            np.add.at(totalFeatures, target.data.cpu().numpy(), 1)

        
        class_means = class_means / totalFeatures
        
        # compute precision
        covariance = np.zeros((model.featureSize, model.featureSize),dtype=np.float32)
        
        for data, y, target in train_loader:
            data, target = data.cuda(), target.cuda()
            _, features = model.forward(data, feature_return=True)
            
            if tasknum > 0:
                
                data_r, y_r, target_r = train_loader.dataset.sample_exemplar()
                data_r, target_r = data_r.cuda(), target_r.cuda()

                _,_,data_r_feature = model.forward(data_r, sample=True)

                features = torch.cat((features, data_r_feature))
                target = torch.cat((target,target_r))
            
            featuresNp = features.data.cpu().numpy()
            vec = featuresNp - class_means[target.data.cpu().numpy()]
            np.expand_dims(vec, axis=2)
            cov = np.matmul(np.expand_dims(vec, axis=2), np.expand_dims(vec, axis=1)).sum(axis=0)
            covariance += cov
        
        #avoid singular matrix
        covariance = covariance / totalFeatures.sum() + np.eye(model.featureSize, dtype=np.float32) * 1e-9
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
            # features: NxD
            # features - mean: NxC(start~end)xD
            start = 0
            if mode == 'train':
                start = tasknum * step_size
                target = target%step_size
            end = (tasknum+1) * step_size
            
#             batch_vec = (features.unsqueeze(1) - self.class_means[start:end].unsqueeze(0)).view(-1,features.shape[1])
            batch_vec = (features.unsqueeze(1) - self.class_means[start:end].unsqueeze(0))
            temp = torch.matmul(batch_vec, self.precision)
            Mahalanobis = torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3))
            _, pred = torch.min(Mahalanobis,1)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            self.class_means = self.class_means.squeeze()

        return 100. * correct / len(loader.dataset)

class softmax_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self):
        pass

    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :return: 
        '''
        model.eval()
        correct = 0
        correct_5 = 0
        total = 0
        self.start = start
        self.end = end
        self.step_size = step_size
        self.stat = {}
        self.correct = {}
        self.correct_5 = {}
        self.correct['bin'] = 0
        self.correct_5['bin'] = 0
        self.correct['cheat'] = 0
        self.correct_5['cheat'] = 0
        head_arr = ['all', 'prev_new', 'task']
        for head in head_arr:
            # cp, epp, epn, cn, enn, enp, total
            self.stat[head] = [0,0,0,0,0,0,0]
            self.correct[head] = 0
            self.correct_5[head] = 0
        
        
        self.bin_target_arr = []
        self.bin_prob_arr = []
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            
            self.batch_size = data.shape[0]
            total += data.shape[0]

            
            if mode == 'test' and end > step_size:
                bin_target = target.data.cpu().numpy() >= (end-step_size)

                out, bin_out = model(data, bc=True)
                
                bin_out = F.softmax(out[:,end-step_size:end],dim=1).data.max(1, keepdim=True)[0].squeeze()
                
                self.bin_cnt(bin_out, bin_target)

                self.make_pred(out)
                
                if target[0]<end-step_size: # prev
                    
                    self.cnt_stat(target, 'prev', 'all')
                    self.cnt_stat(target, 'prev', 'prev_new')
                    self.cnt_stat(target, 'prev', 'task')
                    
                    output = out[:,start:end-step_size]
                    target = target % (end - start-step_size)
                    
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    ans = pred.eq(target.data.view_as(pred)).cpu().sum()
                    self.correct['cheat'] += ans
                    
                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()
                    self.correct_5['cheat'] += ans
                    
                else: # new
                    
                    self.cnt_stat(target, 'new', 'all')
                    self.cnt_stat(target, 'new', 'prev_new')
                    self.cnt_stat(target, 'new', 'task')
                    
                    output = out[:,end-step_size:end]
                    target = target % (step_size)
                    
                    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    ans = pred.eq(target.data.view_as(pred)).cpu().sum()
                    self.correct['cheat'] += ans
                    
                    pred_5 = torch.topk(output, 5, dim=1)[1]
                    ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()
                    self.correct_5['cheat'] += ans
            else:
                output = model(data)[:,start:end]
                target = target % (end - start)
            
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                pred_5 = torch.topk(output, 5, dim=1)[1]
                correct_5 += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()

        if mode == 'test' and end > step_size:
            bin_target = np.hstack(self.bin_target_arr)
            bin_prob = np.hstack(self.bin_prob_arr)
#             auroc = roc_auc_score(bin_target, bin_prob)
        
            for head in ['all','prev_new','task','cheat','bin']:
                self.correct[head] = 100. * self.correct[head] / total
                self.correct_5[head] = 100. * self.correct_5[head] / total
            self.stat['all'][6] = self.stat['prev_new'][6] = self.stat['task'][6] = total
            return self.correct, self.correct_5, self.stat, bin_target, bin_prob

        return 100. * correct / total, 100. * correct_5 / total, 
    
    def bin_cnt(self, bin_out, bin_target):
#         bin_prob = F.sigmoid(bin_out).squeeze()
        bin_prob = bin_out.data.cpu().numpy()
    
        bin_pred = bin_prob > 0.5

        self.bin_target_arr.append(bin_target)
        self.bin_prob_arr.append(bin_prob)
        
        return
    
    def make_pred(self, out):
        start, end, step_size = self.start, self.end, self.step_size
        self.pred = {}
        self.pred_5 = {}
        self.pred['all'] = out.data.max(1, keepdim=True)[1]
        self.pred_5['all'] = torch.topk(out, 5, dim=1)[1]
        
        prev_out = out[:,start:end-step_size]
        curr_out = out[:,end-step_size:end]

        prev_soft = F.softmax(prev_out, dim=1)
        curr_soft = F.softmax(curr_out, dim=1)

        output = torch.cat((prev_soft, curr_soft), dim=1)

        self.pred['prev_new'] = output.data.max(1, keepdim=True)[1]
        self.pred_5['prev_new'] = torch.topk(output, 5, dim=1)[1]
        
        soft_arr = []
        for t in range(start,end,step_size):
            temp_out = out[:,t:t+step_size]
            temp_soft = F.softmax(temp_out, dim=1)
            soft_arr += [temp_soft]
        
        output = torch.cat(soft_arr, dim=1)
        
        self.pred['task'] = output.data.max(1, keepdim=True)[1]
        self.pred_5['task'] = torch.topk(output, 5, dim=1)[1]
        return
    
    def cnt_stat(self, target, mode, head):
        start, end, step_size = self.start, self.end, self.step_size
        pred = self.pred[head]
        pred_5 = self.pred_5[head]
        self.correct[head] += pred.eq(target.data.view_as(pred)).cpu().sum()
        self.correct_5[head] += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()
        
        if mode == 'prev':
            cp_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            epn_ = (pred.cpu().numpy() >= end-step_size).sum()
            epp_ = (self.batch_size-(cp_ + epn_))
            self.stat[head][0] += cp_
            self.stat[head][1] += epp_
            self.stat[head][2] += epn_
        else:
            cn_ = pred.eq(target.data.view_as(pred)).cpu().sum()
            enp_ = (pred.cpu().numpy() < end-step_size).sum()
            enn_ = (self.batch_size-(cn_ + enp_))
            self.stat[head][3] += cn_
            self.stat[head][4] += enn_
            self.stat[head][5] += enp_
        return
    
class sigmoid_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self):
        pass

    def evaluate(self, model, loader, start, end, step_size=100):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :return: 
        '''
        model.eval()
        correct = 0
        correct_5 = 0
        total = 0
        self.start = start
        self.end = end
        self.step_size = step_size
        
        self.bin_target_arr = []
        self.bin_prob_arr = []
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            
            self.batch_size = data.shape[0]
            total += data.shape[0]
            
            bin_target = target.data.cpu().numpy() >= (end-step_size)
            _, bin_out = model(data, bc=True)
            self.bin_cnt(bin_out, bin_target)

        bin_target = np.hstack(self.bin_target_arr)
        bin_prob = np.hstack(self.bin_prob_arr)
        
        return bin_target, bin_prob
    
    def bin_cnt(self, bin_out, bin_target):
        bin_prob = torch.sigmoid(bin_out).squeeze().data.cpu().numpy()
        self.bin_target_arr.append(bin_target)
        self.bin_prob_arr.append(bin_prob)
        
        return