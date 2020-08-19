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

#loader = dataset.loader
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
                                                     #loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     approach = args.trainer
                                                     )

kwargs = {'num_workers': args.workers, 'pin_memory': True}
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=1000, shuffle=False, **kwargs)
myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()


start = 0
end = args.step_size

target_num = 5
feature_result = torch.zeros((2, target_num*2, 1000, 64))

model_name = '200819_EEIL_30_CIFAR10_eeil_0_memsz_2000_base_5_step_5_batch_128_epoch_40_task'
#model_name = '200819_BiC_CIFAR10_bic_0_memsz_2000_base_5_step_5_batch_128_epoch_250_distill_kd_task'
#model_name = '0810WA_Imagenet_wa_0_memsz_20000_base_100_step_100_batch_256_epoch_100_task'


for t in range(2):
    
    name = 'models/trained_model/' + model_name + '_%d.pt'
    #if t == 0:
    #    test_iterator.dataset.task_change()
    #    start = end
    #    end += args.step_size
    #    continue

    index = torch.zeros(10).long()
    myModel.load_state_dict(torch.load(name%t))
    
    old_class = torch.randperm(start)[:target_num]
    new_class = torch.randperm(args.step_size)[:target_num] + start

    target_class = {}
    if t != 0:
        for i in range(target_num):
            target_class[old_class[i].item()] = i
        for i in range(target_num, 2*target_num):
            target_class[new_class[i-target_num].item()] = i
    else:
        for i in range(target_num):
            target_class[new_class[i].item()] = i
    print(target_class) 
    #target_class = {0:0, 1:1, 2:2, 100:3, 101:4, 102:5}
        
    for data, target in tqdm(test_iterator):
        data, target = data.cuda(), target.cuda()
        
        if target[0].item() not in target_class:
            continue
        with torch.no_grad():
            _, feature = myModel(data, feature_return=True)
            for i in range(len(target)):
                feature_result[0][target[i]][index[target[i]]] = feature[i]
                index[target[i]] += 1
        print(index)       
            #feature_result[t][target_class[target[0].item()]] = feature
    
    test_iterator.dataset.task_change()
    start = end
    end += args.step_size
    print(index)
    #print(feature_result[t-1])
    #feature_result_s = feature_result.cpu().numpy() 
    #np.save("feature_result/{}".format(model_name)+"_{}".format(target_num), feature_result_s)

feature_result = feature_result.cpu().numpy()
#print(feature_result)
np.save("feature_result/{}".format(model_name)+"_{}".format(target_num), feature_result)
