import torch
class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset):
        
        if dataset == 'CIFAR100':
            
            import networks.resnet32 as res
            return res.resnet32(100)
        
        if dataset == 'CIFAR10':
            
            import networks.resnet18 as res
            return res.resnet18(10)
        
        elif dataset == 'Imagenet' or dataset == 'VggFace2_1K' or dataset == 'Google_Landmark_v2_1K':
            
            import networks.resnet18 as res
            return res.resnet18(1000)
        
        elif dataset == 'VggFace2_5K':\
            
            import networks.resnet18 as res
            return res.resnet18(5000)
        
        elif dataset == 'Google_Landmark_v2_10K':
            
            import networks.resnet18 as res
            return res.resnet18(10000)
