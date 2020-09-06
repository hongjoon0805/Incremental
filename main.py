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

# Iterator to iterate over training data.
kwargs = {'num_workers': args.workers, 'pin_memory': True}

incremental_loader.mode = 'train'
train_iterator = torch.utils.data.DataLoader(incremental_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

# Iterator to iterate over test data
incremental_loader.mode = 'test'
test_iterator = torch.utils.data.DataLoader(incremental_loader, batch_size=100, shuffle=False, **kwargs)

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=False)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, myModel, args, optimizer)

print(args.step_size)

schedule = np.array(args.schedule)

tasknum = (dataset.classes-args.base_classes)//args.step_size+1
total_epochs = args.nepochs

print(tasknum)

# initialize result logger
logger = trainer.ResultLogger(myTrainer, train_iterator, test_iterator, args)
logger.make_log_name()

for t in range(tasknum):
    
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr
    if args.trainer == 'ssil' or 'ft' in  args.trainer:
        lr = args.lr / (t+1)
        if t==1:
            total_epochs = args.nepochs // args.factor
            schedule = schedule // args.factor
    
    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)
    
    # Running nepochs epochs
    for epoch in range(0, total_epochs):
        
        myTrainer.update_lr(epoch, schedule)
        if args.trainer == 'il2m':
            break
        else:
            incremental_loader.mode = 'train'
            myTrainer.train(epoch)
            
        if epoch % 2 == (1 - 1) and args.debug:
            if args.trainer == 'icarl' or 'nem' in args.trainer:
                logger.update_moment()
            logger.evaluate(mode='train', get_results = False)
            logger.evaluate(mode='test', get_results = False)
    
    # iCaRL prototype update
    if args.trainer == 'icarl' or 'nem' in args.trainer:
        logger.update_moment()
        print('Moment update finished')
    
    # IL2M mean update
    elif args.trainer == 'il2m':
        logger.update_mean()
        print('Mean update finished')
    
    # WA weight align
    if t > 0 and (args.trainer == 'ft_wa' or args.trainer == 'wa'):
        myTrainer.weight_align()   
    
    # EEIL balanced fine-tuning
    if t > 0 and args.trainer == 'eeil':
        myTrainer.update_frozen_model()
        myTrainer.balance_fine_tune()
    
    ############################################
    #        BIC bias correction train         #
    ############################################
    
    if 'bic' in args.trainer and t>0:
        incremental_loader.mode = 'bias'
        
        for e in range(total_epochs*2):
            myTrainer.train_bias_correction(train_iterator)
            myTrainer.update_bias_lr(e, schedule)
            
    logger.evaluate(mode='train', get_results = False)
    logger.evaluate(mode='test', get_results = True)
    
    start = 0
    end = args.base_classes
    result_loader.reset()
    iterator = torch.utils.data.DataLoader(result_loader, batch_size=100, **kwargs)
    for i in range(t+1):
        logger.get_task_accuracy(start, end, iterator)
        
        start = end
        end += args.step_size
        
        result_loader.task_change()
    
    
    myTrainer.increment_classes()
    logger.save_results()
