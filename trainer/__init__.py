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
    def __init__(self, trainer, incremental_loader, args):
        
        self.tasknum = (incremental_loader.classes-args.base_classes)//args.step_size+1
        
        self.trainer = trainer
        self.model = trainer.model
        self.incremental_loader = incremental_loader
        self.args = args
        self.kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.option = 'Euclidean'
        self.result = {}
        
        # For IL2M
        classes = incremental_loader.classes
        self.init_class_means = torch.zeros(classes).cuda()
        self.current_class_means = torch.zeros(classes).cuda()
        self.model_confidence = torch.zeros(classes).cuda()
        
    def evaluate(self, mode = 'train', get_results = False):
        self.model.eval()
        t = self.incremental_loader.t
        if mode == 'train':
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            
        elif mode == 'test':
            self.incremental_loader.mode = 'test'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=100, shuffle=False, **self.kwargs)
        
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
                
            out = torch.cat(out_matrix, dim=0)
            features = torch.cat(features_matrix, dim=0)
            target = torch.cat(target_matrix, dim=0)
            self.get_accuracy(mode, out, target)
            
            if mode == 'test' and get_results and t > 0:
                self.get_statistics(out, target)
                self.get_confusion_matrix(out, target)
#                 self.get_cosine_similarity_score_average(out, features, target)
                self.get_weight_norm()
#                 self.get_features_norm(features)

            self.print_result(mode, t)
        
    def make_output(self, data):
        end = self.incremental_loader.end
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
            
        elif self.args.trainer == 'icarl' or 'nem' in self.args.trainer or self.args.trainer == 'gda':
            _, features = self.model.forward(data, feature_return=True)
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
            
        t = self.incremental_loader.t
        
        pred_1 = out.data.max(1, keepdim=True)[1]
        pred_5 = torch.topk(out, 5, dim=1)[1]

        correct_1 = pred_1.eq(target.data.view_as(pred_1)).sum().item()
        correct_5 = pred_5.eq(target.data.unsqueeze(1).expand(pred_5.shape)).sum().item()
        
        self.result[mode+'-top-1'][t] = round(100.*(correct_1 / target.shape[0]), 2)
        self.result[mode+'-top-5'][t] = round(100.*(correct_5 / target.shape[0]), 2)
        
        
    def get_statistics(self, out, target):
        if 'statistics' not in self.result:
            self.result['statistics'] = []
        
        stat = [0,0,0,0,0,0]
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
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
        
    def get_cosine_similarity_score_softmax_average(self, out, features, target):
        # ground truth 와 prediction 사이의 cosine similarity
        if 'cosine_similarity' not in self.result:
            self.result['score'] = {}
            self.result['softmax'] = {}
            self.result['cosine_similarity'] = {}
            
            self.result['score']['old_class_pred'] = []
            self.result['score']['new_class_pred'] = []
            self.result['score']['epn'] = []
            self.result['score']['enp'] = []
            
            self.result['softmax']['old_class_pred'] = []
            self.result['softmax']['new_class_pred'] = []
            self.result['softmax']['epn'] = []
            self.result['softmax']['enp'] = []
            
            self.result['cosine_similarity']['old_class_pred'] = []
            self.result['cosine_similarity']['new_class_pred'] = []
            self.result['cosine_similarity']['epn'] = []
            self.result['cosine_similarity']['enp'] = []
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
        mid = end - self.args.step_size
        samples_per_classes = target.shape[0] // end
        old_samples = (end - self.args.step_size) * samples_per_classes 
        
        weight = self.model.module.fc.weight
        sample_size = out.shape[0]
        pred = out.data.max(1, keepdim=True)[1]
        normalized_features = features / torch.norm(features, 2, 1).unsqueeze(1)
        normalized_weight = weight / torch.norm(weight, 2, 1).unsqueeze(1)
        cos_sim_matrix = torch.matmul(normalized_features, normalized_weight.transpose(0,1))
        
        softmax = F.softmax(out, dim=1)
        
        old_class_pred, new_class_pred = pred < mid, pred >= mid
        
        old_score_avg, new_score_avg = out[old_class_pred].mean(), out[new_class_pred].mean()
        old_softmax_avg, new_softmax_avg = softmax[old_class_pred].mean(), softmax[new_class_pred].mean()
        old_cos_sim_avg, new_cos_sim_avg = cos_sim_matrix[old_class_pred].mean(), cos_sim_matrix[new_class_pred].mean()
        
        epn_mask, enp_mask = pred[:old_samples] >= mid, pred[old_samples:] < mid
        
        epn_score_avg, enp_score_avg = out[:old_samples][epn_mask].mean(), out[old_samples:][enp_mask].mean()
        epn_softmax_avg, enp_softmax_avg = softmax[:old_samples][epn_mask].mean(), softmax[old_samples:][enp_mask].mean()
        epn_cos_sim_avg = cos_sim_matrix[:old_samples][epn_mask].mean()
        enp_cos_sim_avg = cos_sim_matrix[old_samples:][enp_mask].mean()
        
        self.result['score']['old_class_pred'].append(old_score_avg)
        self.result['score']['new_class_pred'].append(new_score_avg)
        self.result['score']['epn'].append(epn_score_avg)
        self.result['score']['enp'].append(enp_score_avg)
        
        self.result['softmax']['old_class_pred'].append(old_softmax_avg)
        self.result['softmax']['new_class_pred'].append(new_softmax_avg)
        self.result['softmax']['epn'].append(epn_softmax_avg)
        self.result['softmax']['enp'].append(enp_softmax_avg)
        
        self.result['cosine_similarity']['old_class_pred'].append(old_cos_sim_avg)
        self.result['cosine_similarity']['new_class_pred'].append(new_cos_sim_avg)
        self.result['cosine_similarity']['epn'].append(epn_cos_sim_avg)
        self.result['cosine_similarity']['enp'].append(enp_cos_sim_avg)
        
        return
        
    def get_weight_norm(self):
        # get average weight norm of model
        if 'weight_norm' not in self.result:
            self.result['weight_norm'] = []
        
        end = self.incremental_loader.end
        weight = self.model.module.fc.weight
        norm = torch.norm(weight, 2, 1).unsqueeze(1)
        
        self.result['weight_norm'].append(norm[:end].data.cpu().numpy())
        
        return
        
    def get_features_norm(self, features):
        if 'features_norm' not in self.result:
            self.result['features_norm'] = {}
            self.result['features_norm']['old'] = []
            self.result['features_norm']['new'] = []
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
        samples_per_classes = target.shape[0] // end
        old_samples = (end - self.args.step_size) * samples_per_classes 
        
        norm = torch.norm(features, 2, 1).unsqueeze(1)
        
        norm_old = norm[:old_samples].mean()
        norm_new = norm[old_samples:].mean()
        
        self.result['features_norm']['old'].append(norm_old.data.cpu().numpy())
        self.result['features_norm']['new'].append(norm_new.data.cpu().numpy())
        
        return
    
    def get_entropy(self, out, target):
        if 'entropy' not in self.result:
            self.result['entropy'] = {}
            self.result['entropy']['old'] = []
            self.result['entropy']['new'] = []
        
        
        t = self.incremental_loader.t
        end = self.incremental_loader.end
        samples_per_classes = target.shape[0] // end
        old_samples = (end - self.args.step_size) * samples_per_classes 
        
        prob = F.softmax(out, dim=1)
        log_prob = F.log_softmax(out, dim=1)
        
        old_entropy = (-log_prob[:old_samples] * prob[:old_samples]).sum(dim=1).mean()
        new_entropy = (-log_prob[old_samples:] * prob[old_samples:]).sum(dim=1).mean()
        
        self.result['entropy']['old'].append(old_entropy)
        self.result['entropy']['new'].append(new_entropy)
        
        pass
    
    
    def get_task_accuracy(self, start, end, t, iterator):
        if 'task_accuracy' not in self.result:
            self.result['task_accuracy'] = np.zeros((self.tasknum, self.tasknum))
        
        with torch.no_grad():
        
            out_matrix = []
            target_matrix = []
            for data, target in tqdm(iterator):
                data, target = data.cuda(), target.cuda()
                target = target % (end-start)
                out = self.model(data)[:,start:end]
                out_matrix.append(out)
                target_matrix.append(target)

            out = torch.cat(out_matrix)
            target = torch.cat(target_matrix)
            
            pred = out.data.max(1, keepdim=True)[1]

            correct = pred.eq(target.data.view_as(pred)).sum().item()
            task = iterator.dataset.t
            self.result['task_accuracy'][t][task] = 100.*(correct / target.shape[0])
            
        return
    
    def update_moment(self):
        self.model.eval()
        
        tasknum = self.incremental_loader.t
        with torch.no_grad():
            # compute means
            classes = self.args.step_size * (tasknum+1)
            class_means = torch.zeros((classes,512)).cuda()
            totalFeatures = torch.zeros((classes, 1)).cuda()
            total = 0
            
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            for data, target in tqdm(iterator):
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
                for data, target in tqdm(iterator):
                    data, target = data.cuda(), target.cuda()
                    _, features = self.model.forward(data, feature_return=True)

                    vec = (features.data - class_means[target])
                    
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
        classes = self.incremental_loader.classes
        end = self.incremental_loader.end
        step_size = self.args.step_size
        with torch.no_grad():
            class_means = torch.zeros(classes).cuda()
            class_count = torch.zeros(classes).cuda()
            current_count = 0
            
            self.incremental_loader.mode = 'evaluate'
            iterator = torch.utils.data.DataLoader(self.incremental_loader,
                                                   batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            for data, target in tqdm(iterator):
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
        
        if self.args.bft:
            self.log_name += '_bft_lr_{}'.format(self.args.bft_lr)
        
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
        t = self.incremental_loader.t
        torch.save(self.model.state_dict(), './models/trained_model/' + self.log_name + '_task_{}.pt'.format(t))
        if 'bic' in self.args.trainer:
            torch.save(self.trainer.bias_correction_layer.state_dict(), 
                       './models/trained_model/' + self.log_name + '_bias' + '_task_{}.pt'.format(t))
                
