import argparse

import torch
import torch.utils.data as td
import numpy as np
import scipy.io as sio
import copy

import data_handler
import networks
import trainer
import arguments
# import deepspeed
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

args = arguments.get_args()

dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
dataset.shuffle_data(shuffle_idx)
print("Label shuffled")
print(shuffle_idx)

myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()

incremental_loader = data_handler.IncrementalLoader(dataset, args)
result_loader = data_handler.ResultLoader(dataset, args)
result_loader.reset()

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(incremental_loader, myModel, args)

schedule = np.array(args.schedule)
tasknum = (dataset.classes-args.base_classes)//args.step_size+1
total_epochs = args.nepochs

# initialize result logger
logger = trainer.ResultLogger(myTrainer, incremental_loader, args)
logger.make_log_name()

for t in range(tasknum):
    
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr
    if args.trainer == 'ssil' or 'ft' in  args.trainer:
        lr = args.lr / (t+1)
        if t==1:
            total_epochs = args.nepochs // args.factor
            schedule = schedule //  args.factor
    
    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)
    flag = 0
    if (args.trainer == 'ft' or args.trainer == 'ssil') and t==0:
        name = 'models/trained_model/200729_FT_BIC_Imagenet_ft_bic_0_memsz_20000_base_100_step_100_batch_128_epoch_100_task_0.pt'
        state_dict = torch.load(name)
        myTrainer.model.load_state_dict(state_dict)
        flag = 1
    
    if args.trainer == 'gda':
        try:
            name = 'models/trained_model/GDA_Eeuclidean_Imagenet_gda_0_memsz_20000_base_100_step_100_batch_128_epoch_100_task_%d.pt'%(t+1)
            state_dict = torch.load(name)
            myTrainer.model.load_state_dict(state_dict)
            flag = 1
        except:
            pass
    # Running nepochs epochs
    if t>0 and args.trainer == 'gda':
        mean = copy.deepcopy(logger.class_means)
        precision = copy.deepcopy(logger.precision)
    for epoch in range(0, total_epochs):
        if flag == 1:
            print('Evaluation!')
            break
        myTrainer.update_lr(epoch, schedule)
        if args.trainer == 'il2m':
            break
        else:
            incremental_loader.mode = 'train'
            if args.trainer == 'gda' and t>0:
                myTrainer.train(epoch, mean, precision)
                
            else:
                myTrainer.train(epoch)
            
        if epoch % 10 == (10 - 1) and args.debug:
            if args.trainer == 'icarl' or 'nem' in args.trainer or args.trainer == 'gda':
                logger.update_moment()
            logger.evaluate(mode='train', get_results = False)
            logger.evaluate(mode='test', get_results = False)
    
    # iCaRL prototype update
    if args.trainer == 'icarl' or 'nem' in args.trainer or args.trainer == 'gda':
        logger.update_moment()
        print('Moment update finished')
    
    # IL2M mean update
    if args.trainer == 'il2m':
        logger.update_mean()
        print('Mean update finished')
    
    # WA weight align
    if t > 0 and 'wa' in args.trainer:
        myTrainer.weight_align()   
    
    # EEIL balanced fine-tuning
    if t > 0 and args.bft:
        logger.save_model(add_name = '_before_bft')
        myTrainer.balance_fine_tune()
        
    
    # BiC Bias correction
    if t > 0 and 'bic' in args.trainer:
        myTrainer.train_bias_correction()
            
    logger.evaluate(mode='train', get_results = False)
    logger.evaluate(mode='test', get_results = True)
    
    start = 0
    end = args.base_classes
    
    result_loader.reset()
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    iterator = torch.utils.data.DataLoader(result_loader, batch_size=100, **kwargs)
    for i in range(t+1):
        logger.get_task_accuracy(start, end, t, iterator)
        
        start = end
        end += args.step_size
        
        result_loader.task_change()
    
    myTrainer.increment_classes()
    logger.save_results()
    logger.save_model()
