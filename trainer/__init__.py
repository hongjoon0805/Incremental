from trainer.evaluator import *
from trainer.trainer_factory import *
from trainer.loss import *

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pickle


class ResultLogger():
    def __init__(self, trainer, train_iterator, test_iterator, args):
        
        self.tasknum = (train_iterator.dataset.classes-args.base_classes)//args.step_size+1
        
        self.trainer = trainer
        self.model = trainer.model
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.args = args
        self.option = 'euclidean'
        self.result = {}
        
        # For IL2M
        classes = self.train_iterator.dataset.dataset.classes
        self.init_class_means = torch.zeros(classes).cuda()
        self.current_class_means = torch.zeros(classes).cuda()
        self.model_confidence = torch.zeros(classes).cuda()
        
    def evaluate(self, mode = 'train', get_results = False):
        self.model.eval()
        t = self.train_iterator.dataset.t
        if mode == 'train':
            self.train_iterator.dataset.mode = 'evaluate'
            iterator = self.train_iterator
        elif mode == 'test':
            self.train_iterator.dataset.mode = 'test'
            iterator = self.test_iterator
        
        with torch.no_grad():
            out_matrix = []
            target_matrix = []
            features_matrix = []
            
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                out, features = self.make_output(data)
                out_matrix.append(out)
                features_matrix.append(features)
                target_matrix.append(target)
                
            out_matrix = torch.cat(out_matrix, dim=0)
            features_matrix = torch.cat(features_matrix, dim=0)
            target_matrix = torch.cat(target_matrix, dim=0)
            self.get_accuracy(mode, out_matrix, target_matrix)
            
            if mode == 'test' and get_results and t > 0:
                self.get_statistics(out_matrix, target_matrix)
                self.get_confusion_matrix(out_matrix, target_matrix)
#                 self.get_cosine_similarity(out_matrix, features_matrix, target_matrix)
#                 self.get_weight_norm()
#                 self.get_features_norm(features_matrix)

            self.print_result(mode, t)
        
    def make_output(self, data):
        end = self.train_iterator.dataset.end
        step_size = self.args.step_size
        
        if 'bic' in self.args.trainer:
            bias_correction_layer = self.trainer.bias_correction_layer
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
            if end > step_size:
                out_new = bias_correction_layer(out[:,end-step_size:end])
                out = torch.cat((out[:,:end-step_size], out_new), dim=1)
            
        elif self.args.trainer == 'il2m':
            
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
            if end > step_size:
                pred = out.data.max(1, keepdim=True)[1]
                mask = (pred >= end-step_size).int()
                prob = F.softmax(out, dim=1)
                rect_prob = prob * (self.init_class_means[:end] / self.current_class_means[:end]) \
                                 * (self.model_confidence[end-1] / self.model_confidence[:end])
                out = (1-mask).float() * prob + mask.float() * rect_prob
            
        elif self.args.trainer == 'icarl' or 'nem' in self.args.trainer:
            _, features = model.forward(data, feature_return=True)
            batch_vec = (features.data.unsqueeze(1) - self.class_means.unsqueeze(0))
            temp = torch.matmul(batch_vec, self.precision)
            out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
            
        else:
            out, features = self.model(data, feature_return = True)
            out = out[:,:end]
        
        return out, features
    
    def get_accuracy(self, mode, out, target):
        if mode+'-top-1' not in self.result:
            self.result[mode + '-top-1'] = np.zeros(self.tasknum)
            self.result[mode + '-top-5'] = np.zeros(self.tasknum)
            
        t = self.train_iterator.dataset.t
        
        pred_1 = out.data.max(1, keepdim=True)[1]
        pred_5 = torch.topk(out, 5, dim=1)[1]

        correct_1 = pred_1.eq(target.data.view_as(pred_1)).sum().item()
        correct_5 = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
        
        self.result[mode+'-top-1'][t] = 100.*(correct_1 / target.shape[0])
        self.result[mode+'-top-5'][t] = 100.*(correct_5 / target.shape[0])
        
        
    def get_statistics(self, out, target):
        if 'statistics' not in self.result:
            self.result['statistics'] = []
        
        stat = np.zeros(6)
        
        t = self.train_iterator.dataset.t
        end = self.train_iterator.dataset.end
        samples_per_classes = target.shape[0] // end
        old_samples = (end - self.args.step_size) * samples_per_classes 
        new_samples = out.shape[0] - old_samples
        out_old, out_new = out[:old_samples], out[old_samples:]
        target_old, target_new = target[:old_samples], target[old_samples:]
        pred_old, pred_new = out_old.data.max(1, keepdim=True)[1], out_new.data.max(1, keepdim=True)[1]
        
        # statistics
        cp = pred_old.eq(target_old.data.view_as(pred_old)).sum()
        epn = (pred_old >= end-self.args.step_size).int().sum()
        epp = (old_samples-(cp + epn))
        cn = pred_new.eq(target_new.data.view_as(pred_new)).sum()
        enp = (pred_new < end-self.args.step_size).int().sum()
        enn = (new_samples-(cn + enp))
        
        stat[0], stat[1], stat[2], stat[3], stat[4], stat[5] = cp, epn, epp, cn, enp, enn 
        
        # save statistics
        self.result['statistics'].append(stat)
        
    def get_confusion_matrix(self, out, target):
        # task specific confusion matrix 구하기
        if 'confusion_matrix' not in self.result:
            self.result['confusion_matrix'] = []
            
        task_pred = out.data.max(1, keepdim=True)[1] // self.args.step_size
        task_target = target // self.args.step_size
        
        matrix = confusion_matrix(task_target.data.cpu().numpy(), task_pred.data.cpu().numpy())
        
        self.result['confusion_matrix'].append(matrix)
        
        
    def get_cosine_similarity(self, out, features, target):
        # ground truth 와 prediction 사이의 cosine similarity
        if 'cosine_similarity' not in self.result:
            self.result['cosine_similarity'] = {}
            self.result['cosine_similarity']['pred'] = []
            self.result['cosine_similarity']['target'] = []
        
        weight = self.model.module.fc.weight
        sample_size = out.shape[0]
        pred = out.data.max(1, keepdim=True)[1]
        normalized_features = features / torch.norm(features, 2, 1).unsqueeze(1)
        normalized_weight = weight / torch.norm(weight, 2, 1).unsqueeze(1)
        cos_sim_matrix = torch.matmul(normalized_features, normalized_weight.transpose(0,1))
        pred_cos_sim = cos_sim_matrix[torch.arange(sample_size), pred].data.cpu().numpy()
        target_cos_sim = cos_sim_matrix[torch.arange(sample_size), target].data.cpu().numpy()
        
        self.result['cosine_similarity']['pred'].append(pred_cos_sim)
        self.result['cosine_similarity']['target'].append(target_cos_sim)
        
        return
        
    def get_weight_norm(self):
        # get average weight norm of model
        if 'weight_norm' not in self.result:
            self.result['weight_norm'] = []
        
        end = self.train_iterator.dataset.end
        weight = self.model.module.fc.weight
        norm = torch.norm(weight, 2, 1).unsqueeze(1)
        self.result['weight_norm'].append(norm[:end].data.cpu().numpy())
        
        return
        
    def get_features_norm(self, features):
        if 'features_norm' not in self.result:
            self.result['features_norm'] = []
        
        norm = torch.norm(features, 2, 1).unsqueeze(1)
        
        self.result['features_norm'].append(norm.data.cpu().numpy())
        
        return
    
    def get_task_accuracy(self, start, end, iterator):
        if 'task_accuracy' not in self.result:
            self.result['task_accuracy'] = np.zeros((self.tasknum, self.tasknum))
        
        with torch.no_grad():
        
            out_matrix = []
            target_matrix = []
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                out = self.model(data)[:,start:end]
                out_matrix.append(out)
                target_matrix.append(target)

            pred = out.data.max(1, keepdim=True)[1]

            correct = pred.eq(target.data.view_as(pred)).sum().item()

            self.result['task_accuracy'] = 100.*(correct / target.shape[0])
            
        return
    
    def update_moment(self):
        self.model.eval()
        
        tasknum = self.train_iterator.dataset.t
        with torch.no_grad():
            # compute means
            classes = self.args.step_size * (tasknum+1)
            class_means = torch.zeros((classes,512)).cuda()
            totalFeatures = torch.zeros((classes, 1)).cuda()
            total = 0
            
            self.train_iterator.dataset.mode = 'evaluate'
            for data, target in tqdm(self.train_iterator):
                data, target = data.cuda(), target.cuda()
                if data.shape[0]<4:
                    continue
                total += data.shape[0]
                try:
                    _, features = self.model.forward(data, feature_return=True)
                except:
                    continue
                    
                class_means.index_add_(0, target, features.data)
                totalFeatures.index_add_(0, target, torch.ones_like(target.unsqueeze(1)).float().cuda())
                
            class_means = class_means / totalFeatures
            
            # compute precision
            covariance = torch.zeros(512,512).cuda()
            euclidean = torch.eye(512).cuda()

            if self.option == 'Mahalanobis':
                for data, target in tqdm(self.train_iterator):
                    data, target = data.cuda(), target.cuda()
                    _, features = self.model.forward(data, feature_return=True)

                    vec = (features.data - class_means[target])
                    
                    np.expand_dims(vec, axis=2)
                    cov = torch.matmul(vec.unsqueeze(2), vec.unsqueeze(1)).sum(dim=0)
                    covariance += cov

                #avoid singular matrix
                covariance = covariance / totalFeatures.sum() + torch.eye(512).cuda() * 1e-9
                precision = covariance.inverse()

            self.class_means = class_means
            if self.option == 'Mahalanobis':
                self.precision = precision
            else:
                self.precision = euclidean
        
        return
        
    def update_mean(self):
        self.model.eval()
        classes = self.train_iterator.dataset.dataset.classes
        end = self.train_iterator.dataset.end
        step_size = self.args.step_size
        with torch.no_grad():
            class_means = torch.zeros(classes).cuda()
            class_count = torch.zeros(classes).cuda()
            current_count = 0
            
            self.train_iterator.dataset.mode = 'evaluate'
            for data, target in tqdm(self.train_iterator):
                data, target = data.cuda(), target.cuda()
                out = self.model(data)
                prob = F.softmax(out[:,:end], dim=1)
                confidence = prob.max(dim=1)[0] * (target >= (end-step_size)).float()
                class_means.index_add_(0, target, prob[torch.arange(data.shape[0]),target])
                class_count.index_add_(0, target, torch.ones_like(target).float().cuda())
                
                self.model_confidence[end-step_size:end] += confidence.sum()
                current_count += (target >= (end-step_size)).float().sum()

            self.init_class_means[end-step_size:end] = class_means[end-step_size:end] / class_count[end-step_size:end]
            self.current_class_means[:end] = class_means[:end] / class_count[:end]
            self.model_confidence[end-step_size:end] /= current_count
    
    def make_log_name(self):
        
        self.log_name = '{}_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
            self.args.date,
            self.args.dataset,
            self.args.trainer,
            self.args.seed,
            self.args.memory_budget,
            self.args.base_classes,
            self.args.step_size,
            self.args.batch_size,
            self.args.nepochs,
        )

        if self.args.distill != 'None':
            self.log_name += '_distill_{}'.format(self.args.distill)
            
        if self.args.trainer == 'ssil':
            self.log_name += '_replay_{}'.format(self.args.replay_batch_size)
            
        if self.args.trainer == 'ssil' or 'ft' in  self.args.trainer or self.args.trainer == 'il2m':
            self.log_name += '_factor_{}'.format(self.args.factor)
            
        if self.args.prev_new:
            self.log_name += '_prev_new'
            
    def print_result(self, mode, t):
        print(mode + " top-1: %0.2f"%self.result[mode + '-top-1'][t])
        print(mode + " top-5: %0.2f"%self.result[mode + '-top-5'][t])
        
        return
        
    def save_results(self):
        
        path = self.log_name + '.pkl'
        with open('result_data/' + path, "wb") as f:
            pickle.dump(self.result, f)
        
    def save_model(self):
        t = self.train_iterator.dataset.t
        torch.save(self.model.state_dict(), './models/trained_model/' + self.log_name + '_task_{}.pt'.format(t))
        if 'bic' in self.args.trainer:
            torch.save(self.trainer.bias_correction_layer.state_dict(), 
                       './models/trained_model/' + self.log_name + '_bias' + '_task_{}.pt'.format(t))
                