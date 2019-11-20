''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''

from torchvision import datasets, transforms
import torch
import numpy as np

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

        train_dataset = datasets.CIFAR100("../../dat", train=True, transform=self.train_transform, download=True)
        self.train_data = train_dataset.train_data
        self.train_labels = train_dataset.train_labels
        test_dataset = datasets.CIFAR100("../../dat", train=False, transform=self.test_transform, download=True)
        self.test_data = test_dataset.test_data
        self.test_labels = test_dataset.test_labels

class Imagenet(Dataset):
    def __init__(self):
        super().__init__(1000, "Imagenet")
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        
        train_data = datasets.ImageFolder("../../dat/Imagenet/train", transform=self.train_transform)
        test_data = datasets.ImageFolder("../../dat/Imagenet/val", transform=self.test_transform)
        self.loader = train_data.loader
        
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        
        for i in range(len(train_data.imgs)):
            path, target = train_data.imgs[i]
            self.train_data.append(path)
            self.train_labels.append(target)
            
        for i in range(len(test_data.imgs)):
            path, target = test_data.imgs[i]
            self.test_data.append(path)
            self.test_labels.append(target)
        
        self.train_data = np.vstack(self.train_data)
        self.test_data = np.vstack(self.test_data)

