import argparse

import torch
import torch.utils.data as td
import numpy as np
import scipy.io as sio

import data_handler
import networks
import trainer
import arguments
# import deepspeed
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

args = arguments.get_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True
dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

loader = dataset.loader
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
print(shuffle_idx)


test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data,
                                                     dataset.test_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'test',
                                                     transform=dataset.test_transform,
                                                     loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     approach = args.trainer
                                                     )

kwargs = {'num_workers': args.workers, 'pin_memory': True}
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=50, shuffle=False, **kwargs)
print(test_iterator, test_itrerator[0])
myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()


start = 0
end = args.step_size

feature_result = torch.zeros((9, 6, 512))

for t in range(10):
    name = 'models/trained_model/200807_FIT_BIC_Imagenet_bic_0_memsz_20000_base_100_step_100_batch_256_epoch_100_distill_feature_l2_task_%d.pt'
    if t == 0:
        test_iterator.dataset.task_change()
        start = end
        end += args.step_size
        continue

    myModel.load_state_dict(torch.load(name%t))
    
    norm_arr = [0,0]
    cnt_arr = [0,0]
    for data, target in tqdm(test_iterator):
        data, target = data.cuda(), target.cuda()
        
        old_class = torch.randperm(start)[:3]
        new_class = torch.randperm(step_size)[:3] + start

        target_class = {}
        for i in range(3):
            target_class[i] = old_class[i].item()
        for i in range(3, 6):
            target_class[i] = new_class[i-3].item()

        #target_class = {0:0, 1:1, 2:2, 100:3, 101:4, 102:5}

        if target[0] not in target_class:
            continue

        with torch.no_grad():
            _, feature = myModel(data, feature_return=True)
            
            feature_result[t-1][target_class[target[0]]] = feature

    test_iterator.dataset.task_change()
    start = end
    end += args.step_size


print(feature_result)

