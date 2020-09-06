import math
import torch
from torch import nn
from scipy.special import binom


class LSoftmaxLoss(nn.Module):

    def __init__(self, margin, model):
        super().__init__()
        self.margin = margin  # m
        self.beta = 0
        self.beta_min = 0
        self.scale = 0.99
        
        self.weight = model.module.fc.weight.data
        # Initialize L-Softmax parameters
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).cuda()  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).cuda()  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).cuda()  # n
        self.signs = torch.ones(margin // 2 + 1).cuda()  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
    
    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target, end_class):
        if self.training:
            assert target is not None
            x, w = input, self.weight.T
            beta = max(self.beta, self.beta_min)
            #print(x.shape, w.shape)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)
            #print(logit[indexes, target])
            logit[indexes, target] = logit_target_updated_beta
            #print(logit_target_updated_beta)
            
            self.beta *= self.scale
            
            #print(logit[:, :end_class].shape, target.shape)
            res = self.loss(logit[:, :end_class], target)
            if torch.isnan(res):
                print(logit[indexes, target])
                print(logit_target_updated_beta)
                for jj in range(999999999999):
                    continue
            return self.loss(logit[:, :end_class], target)
        else:
            assert target is None
            return input.mm(self.weight)