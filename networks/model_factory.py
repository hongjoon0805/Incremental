''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

import networks.resnet32 as res
import networks.test_model as tm


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset="CIFAR100"):
        
        if dataset == 'CIFAR100':
            return res.resnet32(100)
        elif dataset == 'Imagenet':
            return res.resnet18(1000)