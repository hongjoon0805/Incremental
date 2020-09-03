import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == 'CIFAR10':
            return data.CIFAR10()
        elif name == "CIFAR100":
            return data.CIFAR100()
        elif name == "Imagenet":
            return data.Imagenet()
        elif name == "VggFace2_1K":
            return data.VggFace2_1K()
        elif name == "VggFace2_5K":
            return data.VggFace2_5K()
        elif name == "Google_Landmark_v2_1K":
            return data.Google_Landmark_v2_1K()
        elif name == "Google_Landmark_v2_10K":
            return data.Google_Landmark_v2_10K()
