class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset="CIFAR100", ratio=1/512, trainer = 'bayes'):
        
        if dataset == 'CIFAR100':
            if trainer == 'bayes':
                import networks.resnet32_ucl as res
                return res.resnet32(100, ratio)
            else:
                import networks.resnet32 as res
                return res.resnet32(100)
        elif dataset == 'Imagenet':
            if trainer == 'bayes':
                import networks.resnet18_ucl as res
                return res.resnet18(1000, ratio)
            else:
                import networks.resnet18 as res
                return res.resnet18(1000)
