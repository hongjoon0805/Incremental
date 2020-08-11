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
myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()


start = 0
end = args.step_size

target_num = 3
feature_result = torch.zeros((9, target_num*2, 50, 512))

model_name = '0810WA_Imagenet_wa_0_memsz_20000_base_100_step_100_batch_256_epoch_100_task'

for t in range(10):
    
    name = 'models/trained_model/' + model_name + '_%d.pt'
    if t == 0:
        test_iterator.dataset.task_change()
        start = end
        end += args.step_size
        continue

    myModel.load_state_dict(torch.load(name%t))
    
    old_class = torch.randperm(start)[:target_num]
    new_class = torch.randperm(args.step_size)[:target_num] + start

    target_class = {}
    for i in range(target_num):
        target_class[old_class[i].item()] = i
    for i in range(target_num, 2*target_num):
        target_class[new_class[i-target_num].item()] = i
    print(target_class) 
    #target_class = {0:0, 1:1, 2:2, 100:3, 101:4, 102:5}
        
    for data, target in tqdm(test_iterator):
        data, target = data.cuda(), target.cuda()
        
        if target[0].item() not in target_class:
            continue
        with torch.no_grad():
            _, feature = myModel(data, feature_return=True)
            feature_result[t-1][target_class[target[0].item()]] = feature

    test_iterator.dataset.task_change()
    start = end
    end += args.step_size
    #print(feature_result[t-1])
    #feature_result_s = feature_result.cpu().numpy() 
    #np.save("feature_result/{}".format(model_name)+"_{}".format(target_num), feature_result_s)

feature_result = feature_result.cpu().numpy()
print(feature_result)
np.save("feature_result/{}".format(model_name)+"_{}".format(target_num), feature_result)
