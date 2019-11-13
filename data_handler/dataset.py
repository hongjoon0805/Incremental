''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from torchvision import datasets, transforms
import torch
import numpy

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, classes, name):
        self.classes = classes
        self.name = name
        self.train_data = None
        self.test_data = None


class CIFAR100(Dataset):
    def __init__(self):
        super().__init__(100, "CIFAR100")

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])

        self.train_data = datasets.CIFAR100("../../dat", train=True, transform=self.train_transform, download=True)

        self.test_data = datasets.CIFAR100("../../dat", train=False, transform=self.test_transform, download=True)

class Imagenet(Dataset):
    def __init__(self):
        super().__init__(1000, "Imagenet")
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        
        self.train_data = datasets.ImageFolder("../../dat", train=True, transform=self.train_transform)
        self.test_data = datasets.ImageFolder("../../dat", train=False, transform=self.test_transform)


