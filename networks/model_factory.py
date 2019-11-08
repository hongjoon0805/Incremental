class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset, ratio, trainer, out_channels):
        
        if dataset == 'CIFAR100':
            if trainer == 'bayes':
                import networks.resnet32_ucl as res
                return res.resnet32(100, ratio)
            elif trainer == 'gda':
                import networks.resnet32_gda as res
                return res.resnet32(100, ratio)
            else:
                import networks.resnet32 as res
                return res.resnet32(100, out_channels)
        elif dataset == 'Imagenet':
            if trainer == 'bayes':
                import networks.resnet18_ucl as res
                return res.resnet18(1000, ratio)
            else:
                import networks.resnet18 as res
                return res.resnet18(1000)
