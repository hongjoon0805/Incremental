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

def make_pred(out, start, end, step_size):
    pred = {}
    pred_5 = {}
    pred['all'] = out.data.max(1, keepdim=True)[1]
    pred_5['all'] = torch.topk(out, 5, dim=1)[1]

    prev_out = out[:,start:end-step_size]
    curr_out = out[:,end-step_size:end]

    prev_soft = F.softmax(prev_out, dim=1)
    curr_soft = F.softmax(curr_out, dim=1)

    output = torch.cat((prev_soft, curr_soft), dim=1)

    pred['prev_new'] = output.data.max(1, keepdim=True)[1]
    pred_5['prev_new'] = torch.topk(output, 5, dim=1)[1]

    soft_arr = []
    for t in range(start,end,step_size):
        temp_out = out[:,t:t+step_size]
        temp_soft = F.softmax(temp_out, dim=1)
        soft_arr += [temp_soft]

    output = torch.cat(soft_arr, dim=1)

    pred['task'] = output.data.max(1, keepdim=True)[1]
    pred_5['task'] = torch.topk(output, 5, dim=1)[1]
    
    return pred, pred_5

def cnt_stat(target, start, end, step_size, mode, head, pred, pred_5, correct, correct_5, stat, batch_size):
    
    correct[head] += pred[head].eq(target.data.view_as(pred[head])).cpu().sum()
    correct_5[head] += pred_5[head].eq(target.data.unsqueeze(1).expand(pred_5[head].shape)).cpu().sum()

    if mode == 'prev':
        cp_ = pred[head].eq(target.data.view_as(pred[head])).cpu().sum()
        epn_ = (pred[head].cpu().numpy() >= end-step_size).sum()
        epp_ = (batch_size-(cp_ + epn_))
        stat[head][0] += cp_
        stat[head][1] += epp_
        stat[head][2] += epn_
    else:
        cn_ = pred[head].eq(target.data.view_as(pred[head])).cpu().sum()
        enp_ = (pred[head].cpu().numpy() < end-step_size).sum()
        enn_ = (batch_size-(cn_ + enp_))
        stat[head][3] += cn_
        stat[head][4] += enn_
        stat[head][5] += enp_
    return

def cheat(out, target, start, end, mod, correct, correct_5):
    output = out[:,start:end]
    target = target % (mod)
    
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    ans = pred.eq(target.data.view_as(pred)).cpu().sum()
    correct['cheat'] += ans

    pred_5 = torch.topk(output, 5, dim=1)[1]
    ans = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()
    correct_5['cheat'] += ans

class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''

    def __init__(self):
        pass

    @staticmethod
    def get_evaluator(testType="trainedClassifier", classes=1000, option='euclidean'):
        if testType == "trainedClassifier":
            return softmax_evaluator()
        if testType == "IL2M":
            return IL2M_evaluator(classes=classes)
        if testType == "generativeClassifier":
            return GDA(classes, option = option)

class GDA():

    def __init__(self, classes, option = 'euclidean'):
        self.classes = classes
        self.option = option
    
    def update_moment(self, model, train_loader, step_size, tasknum):
        
        # compute means
        classes = step_size * (tasknum+1)
        class_means = np.zeros((classes,512), dtype=np.float32)
        totalFeatures = np.zeros((classes, 1), dtype=np.float32)
        
        # Iterate over all train Dataset
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            _, features = model.forward(data, feature_return=True)
            
            featuresNp = features.data.cpu().numpy()
            np.add.at(class_means, target.data.cpu().numpy(), featuresNp)
            np.add.at(totalFeatures, target.data.cpu().numpy(), 1)

        
        class_means = class_means / totalFeatures
        
        # compute precision
        covariance = np.zeros((512, 512), dtype=np.float32)
        euclidean = np.eye(512, dtype=np.float32)
        
        if self.option == 'Mahalanobis':
            for data,  target in train_loader:
                data, target = data.cuda(), target.cuda()
                _, features = model.forward(data, feature_return=True)

                featuresNp = features.data.cpu().numpy()
                vec = featuresNp - class_means[target.data.cpu().numpy()]
                np.expand_dims(vec, axis=2)
                cov = np.matmul(np.expand_dims(vec, axis=2), np.expand_dims(vec, axis=1)).sum(axis=0)
                covariance += cov

            #avoid singular matrix
            covariance = covariance / totalFeatures.sum() + np.eye(512, dtype=np.float32) * 1e-9
            precision = inv(covariance)
        
        self.class_means = torch.from_numpy(class_means).cuda()
        if self.option == 'Mahalanobis':
            self.precision = torch.from_numpy(precision).cuda()
        else:
            self.precision = torch.from_numpy(euclidean).cuda()
        
        return
    
    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        
        model.eval()
        correct_cnt = 0
        correct_5_cnt = 0
        total = 0
        stat = {}
        correct = {}
        correct_5 = {}
        correct['cheat'] = 0
        correct_5['cheat'] = 0
        head_arr = ['all', 'prev_new', 'task']
        for head in head_arr:
            # cp, epp, epn, cn, enn, enp, total
            stat[head] = [0,0,0,0,0,0,0]
            correct[head] = 0
            correct_5[head] = 0
        
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            _, features  = model(data, feature_return=True)
            
            batch_size = data.shape[0]
            total += data.shape[0]
            
            # M_distance: NxC(start~end)
            # features: NxD
            # features - mean: NxC(start~end)xD
            
            batch_vec = (features.unsqueeze(1) - self.class_means.unsqueeze(0))
            temp = torch.matmul(batch_vec, self.precision)
            out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
            self.class_means = self.class_means.squeeze()
            
            if mode == 'test' and end > step_size:
                pred, pred_5 = make_pred(out, start, end, step_size)
                
                if target[0]<end-step_size: # prev
                    
                    cnt_stat(target,start,end,step_size,'prev','all',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'prev','prev_new',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'prev','task',pred, pred_5,correct,correct_5,stat,batch_size)
                    
                    cheat(out, target, start, end-step_size, end-start-step_size, correct, correct_5)
                    
                else: # new
                    
                    cnt_stat(target,start,end,step_size,'new','all',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'new','prev_new',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'new','task',pred, pred_5,correct,correct_5,stat,batch_size)
                    
                    cheat(out, target, end-step_size, end, step_size, correct, correct_5)
                    
            else:
                output = out[:,start:end]
                target = target % (end - start)
            
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct_cnt += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                pred_5 = torch.topk(output, 5, dim=1)[1]
                correct_5_cnt += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()

        if mode == 'test' and end > step_size:
        
            for head in ['all','prev_new','task','cheat']:
                correct[head] = 100. * correct[head] / total
                correct_5[head] = 100. * correct_5[head] / total
            stat['all'][6] = stat['prev_new'][6] = stat['task'][6] = total
            return correct, correct_5, stat

        return 100. * correct_cnt / total, 100. * correct_5_cnt / total, 
    
class IL2M_evaluator():
    '''
    Evaluator class for softmax classification 
    '''

    def __init__(self, classes=1000, total_task=10):
        
        self.classes = classes
        self.total_task = total_task
        
        self.init_class_means = np.zeros(classes)
        
        self.current_class_means = np.zeros(classes)
        
        self.model_confidence = np.zeros(total_task)
        self.model_count = np.zeros(total_task)
        
    def update_mean(self, model, loader, end, step_size, tasknum):
        class_means = np.zeros(self.classes)
        class_count = np.zeros(self.classes)
        
        
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            out = model(data)
            prob = F.softmax(out[:,:end], dim=1).data.cpu().numpy()
            confidence = prob.max(axis=1) * (target >= (end-step_size)).int()
            idx = target.data.cpu().numpy()
            class_means[idx] += prob[:idx]
            class_count[idx] += 1
            
            self.model_confidence[t] += confidence.sum()
            self.model_count[t] += (data.shape[0] - (target < (end-step_size)).int().sum())
            
        self.init_class_means[end-step_size:end] = class_means[end-step_size:end] / class_count[end-step_size:end]
        self.current_class_means = class_means / class_count
        
        self.model_confidence[t] /= self.model_count[t]
        
    def new_state(self, model, loader, end, step_size, tasknum):
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            out = model(data)
            prob = F.softmax(out[:,start:end], dim=1).data.cpu().numpy()
            confidence = prob.max(axis=1)
            idx = target.data.cpu().numpy()
            self.init_class_means[idx] += prob[:idx]
            self.init_class_count[idx] += 1
            
            self.model_confidence[t] += confidence.sum()
            self.model_count[t] += data.shape[0]
            
        self.class_means[end-step_size:end] /= self.class_count[end-step_size:end]
        self.model_confidence[t] /= self.model_count[t]
    
    def evaluate(self, model, loader, start, end, mode='train', step_size=100):
        '''
        :param model: Trained model
        :param loader: Data iterator
        :return: 
        '''
        model.eval()
        correct_cnt = 0
        correct_5_cnt = 0
        total = 0
        step_size = step_size
        stat = {}
        correct = {}
        correct_5 = {}
        correct['cheat'] = 0
        correct_5['cheat'] = 0
        head_arr = ['all', 'prev_new', 'task']
        for head in head_arr:
            # cp, epp, epn, cn, enn, enp, total
            stat[head] = [0,0,0,0,0,0,0]
            correct[head] = 0
            correct_5[head] = 0
        
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            
            batch_size = data.shape[0]
            total += data.shape[0]
            
            out = model(data)
            
            if mode == 'test' and end > step_size:
                pred, pred_5 = make_pred(out, start, end, step_size)
                
                if target[0]<end-step_size: # prev
                    
                    cnt_stat(target,start,end,step_size,'prev','all',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'prev','prev_new',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'prev','task',pred, pred_5,correct,correct_5,stat,batch_size)
                    
                    cheat(out, target, start, end-step_size, end - start-step_size, correct, correct_5)
                    
                else: # new
                    
                    cnt_stat(target,start,end,step_size,'new','all',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'new','prev_new',pred,pred_5,correct,correct_5,stat,batch_size)
                    cnt_stat(target,start,end,step_size,'new','task',pred, pred_5,correct,correct_5,stat,batch_size)
                    
                    cheat(out, target, end-step_size, end, step_size, correct, correct_5)
            else:
                output = out[:,start:end]
                target = target % (end - start)
            
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct_cnt += pred.eq(target.data.view_as(pred)).cpu().sum()
                
                pred_5 = torch.topk(output, 5, dim=1)[1]
                correct_5_cnt += pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).cpu().sum()

        if mode == 'test' and end > step_size:
        
            for head in ['all','prev_new','task','cheat']:
                correct[head] = 100. * correct[head] / total
                correct_5[head] = 100. * correct_5[head] / total
            stat['all'][6] = stat['prev_new'][6] = stat['task'][6] = total
            return correct, correct_5, stat

        return 100. * correct_cnt / total, 100. * correct_5_cnt / total, 
    
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
        self.correct['cheat'] = 0
        self.correct_5['cheat'] = 0
        head_arr = ['all', 'prev_new', 'task']
        for head in head_arr:
            # cp, epp, epn, cn, enn, enp, total
            self.stat[head] = [0,0,0,0,0,0,0]
            self.correct[head] = 0
            self.correct_5[head] = 0
        
        
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            
            self.batch_size = data.shape[0]
            total += data.shape[0]

            if mode == 'test' and end > step_size:

                out = model(data)
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
        
            for head in ['all','prev_new','task','cheat']:
                self.correct[head] = 100. * self.correct[head] / total
                self.correct_5[head] = 100. * self.correct_5[head] / total
            self.stat['all'][6] = self.stat['prev_new'][6] = self.stat['task'][6] = total
            return self.correct, self.correct_5, self.stat

        return 100. * correct / total, 100. * correct_5 / total, 
    
    
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
    
