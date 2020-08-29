from trainer.evaluator import *
from trainer.trainer_factory import *

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from tqdm import tqdm



# class ResultLogger():
#     def __init__(self, trainer, train_iterator, test_iterator, args):
        
#         self.trainer = trainer
#         self.model = trainer.model
#         self.train_iterator = train_iterator
#         self.test_iterator = test_iterator
#         self.args = args
#         self.option = 'euclidean'
#         self.result = {}
#         self.result['train-top-1'] = []
#         self.result['test-top-1'] = []
#         self.result['train-top-5'] = []
#         self.result['test-top-5'] = []
        
#         # For IL2M
#         classes = self.train_iterator.dataset.dataset.classes
#         self.init_class_means = torch.zeros(classes).cuda()
#         self.current_class_means = torch.zeros(classes).cuda()
#         self.model_confidence = torch.zeros(classes).cuda()
        
#     def evaluate(self, mode = 'train', print_log = False, save_results = False):
        
#         if mode == 'train':
#             self.train_iterator.dataset.mode = 'evaluate'
#             for data, target in tqdm(self.train_iterator):
#                 data, target = data.cuda(), target.cuda()
#                 batch_size = data.shape[0]
#                 total += data.shape[0]
                
#                 out = self.make_output(data)
            
#         elif mode == 'test':
#             self.test_iterator.dataset.mode = 'test'
#             for data, target in tqdm(self.test_iterator):
#                 data, target = data.cuda(), target.cuda()
        
#         if args.debug == 1:
#             self.print_result()
        
#     def make_output(self, data):
#         end = self.train_iterator.dataset.end
#         step_size = self.args.step_size
        
#         if 'bic' in args.trainer:
#             bias_correction_layer = self.trainer.bias_correction_layer
#             out = self.model(data)[:,:end]
#             if end > step_size:
#                 out_new = bias_correction_layer(out[:,end-step_size:end])
#                 out = torch.cat((out[:,:end-step_size], out_new), dim=1)
            
#         elif args.trainer == 'il2m':
            
#             out = model(data)[:,:end]
#             if end > step_size:
#                 pred = out.data.max(1, keepdim=True)[1]
#                 mask = (pred >= end-step_size).int()
#                 prob = F.softmax(out, dim=1)
#                 rect_prob = prob * (self.init_class_means[:end] / self.current_class_means[:end]) \
#                                  * (self.model_confidence[end-1] / self.model_confidence[:end])
#                 out = (1-mask).float() * prob + mask.float() * rect_prob
            
#         elif args.trainer == 'icarl' or 'nem' in args.trainer:
#             _, features = model.forward(data, feature_return=True)
#             batch_vec = (features.data.unsqueeze(1) - self.class_means.unsqueeze(0))
#             temp = torch.matmul(batch_vec, self.precision)
#             out = -torch.matmul(temp.unsqueeze(2),batch_vec.unsqueeze(3)).squeeze()
            
#         else:
#             out = self.model(data)[:,:end]
        
#         return out
    
#     def update_moment(self):
#         self.model.eval()
        
#         tasknum = train_iterator.dataset.t
#         with torch.no_grad():
#             # compute means
#             classes = step_size * (tasknum+1)
#             class_means = torch.zeros((classes,512)).cuda()
#             totalFeatures = torch.zeros((classes, 1)).cuda()
#             total = 0
            
#             self.train_iterator.dataset.mode = 'evaluate'
#             for data, target in tqdm(self.train_iterator):
#                 data, target = data.cuda(), target.cuda()
#                 if data.shape[0]<4:
#                     continue
#                 total += data.shape[0]
#                 try:
#                     _, features = self.model.forward(data, feature_return=True)
#                 except:
#                     continue
                    
#                 class_means.index_add_(0, target, features.data)
#                 totalFeatures.index_add_(0, target, torch.ones_like(target.unsqueeze(1)).float().cuda())
                
#             class_means = class_means / totalFeatures
            
#             # compute precision
#             covariance = torch.zeros(512,512).cuda()
#             euclidean = torch.eye(512).cuda()

#             if self.option == 'Mahalanobis':
#                 for data, target in tqdm(self.train_iterator):
#                     data, target = data.cuda(), target.cuda()
#                     _, features = self.model.forward(data, feature_return=True)

#                     vec = (features.data - class_means[target])
                    
#                     np.expand_dims(vec, axis=2)
#                     cov = torch.matmul(vec.unsqueeze(2), vec.unsqueeze(1)).sum(dim=0)
#                     covariance += cov

#                 #avoid singular matrix
#                 covariance = covariance / totalFeatures.sum() + torch.eye(512).cuda() * 1e-9
#                 precision = covariance.inverse()

#             self.class_means = class_means
#             if self.option == 'Mahalanobis':
#                 self.precision = precision
#             else:
#                 self.precision = euclidean
        
#         return
        
#     def update_mean(self):
#         self.model.eval()
#         classes = self.train_iterator.dataset.dataset.classes
#         end = self.train_iterator.dataset.end
#         step_size = self.args.step_size
#         with torch.no_grad():
#             class_means = torch.zeros(classes).cuda()
#             class_count = torch.zeros(classes).cuda()
#             current_count = 0
            
#             self.train_iterator.dataset.mode = 'evaluate'
#             for data, target in tqdm(self.train_iterator):
#                 data, target = data.cuda(), target.cuda()
#                 out = model(data)
#                 prob = F.softmax(out[:,:end], dim=1)
#                 confidence = prob.max(dim=1)[0] * (target >= (end-step_size)).float()
#                 class_means.index_add_(0, target, prob[torch.arange(data.shape[0]),target])
#                 class_count.index_add_(0, target, torch.ones_like(target).float().cuda())
                
#                 self.model_confidence[end-step_size:end] += confidence.sum()
#                 current_count += (target >= (end-step_size)).float().sum()

#             self.init_class_means[end-step_size:end] = class_means[end-step_size:end] / class_count[end-step_size:end]
#             self.current_class_means[:end] = class_means[:end] / class_count[:end]
#             self.model_confidence[end-step_size:end] /= current_count
    
#     def make_log_name(self):
        
#         self.log_name = '{}_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
#             self.args.date,
#             self.args.dataset,
#             self.args.trainer,
#             self.args.seed,
#             self.args.memory_budget,
#             self.args.base_classes,
#             self.args.step_size,
#             self.args.batch_size,
#             self.args.nepochs,
#         )

#         if self.args.distill != 'None':
#             self.log_name += '_distill_{}'.format(self.args.distill)
            
#         if self.args.trainer == 'ssil':
#             self.log_name += '_replay_{}'.format(self.args.replay_batch_size)
            
#         if self.args.trainer == 'ssil' or 'ft' in  self.args.trainer or self.args.trainer == 'il2m':
#             self.log_name += '_factor_{}'.format(self.args.factor)
            
#         if self.args.prev_new:
#             self.log_name += '_prev_new'
            
            
#     def save_results(self):
        
#         path = log_name + '.pkl'
#         with open('result_data/' + path, "wb") as f:
#             pickle.dump(dic, f)
        
#     def save_model(self):
#         t = self.train_iterator.dataset.t
#         torch.save(self.model.state_dict(), './models/trained_model/' + self.log_name + '_task_{}.pt'.format(t))
#         if 'bic' in self.args.trainer:
#             torch.save(self.trainer.bias_correction_layer.state_dict(), 
#                        './models/trained_model/' + self.log_name + '_bias' + '_task_{}.pt'.format(t))
                