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

log_name = '{}_{}_{}_{}_memsz_{}_base_{}_step_{}_batch_{}_epoch_{}'.format(
    args.date,
    args.dataset,
    args.trainer,
    args.seed,
    args.memory_budget,
    args.base_classes,
    args.step_size,
    args.batch_size,
    args.nepochs,
)

if args.distill != 'None':
    log_name += '_distill_{}'.format(args.distill)

if args.trainer == 'ssil':
    log_name += '_replay_{}'.format(args.replay_batch_size)

if args.trainer == 'ssil' or 'ft' in  args.trainer or args.trainer == 'il2m':
    log_name += '_factor_{}'.format(args.factor)
if args.prev_new:
    log_name += '_prev_new'

dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
    loader = None
else:
    loader = dataset.loader
    
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
print("Label shuffled")
print(shuffle_idx)

dataset.train_labels = shuffle_idx[dataset.train_labels]
dataset.test_labels = shuffle_idx[dataset.test_labels]
train_sort_index = np.argsort(dataset.train_labels)
test_sort_index = np.argsort(dataset.test_labels)

dataset.train_labels = dataset.train_labels[train_sort_index]
dataset.test_labels = dataset.test_labels[test_sort_index]
dataset.train_data = dataset.train_data[train_sort_index]
dataset.test_data = dataset.test_data[test_sort_index]


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
                            weight_decay=args.decay, nesterov=True)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, myModel, args, optimizer)


# Initilize the evaluators used to measure the performance of the system.

if args.trainer == 'icarl':
    testType = "generativeClassifier"
elif args.trainer == 'il2m':
    testType = 'il2m'
elif 'bic' in args.trainer:
    testType = 'bic'
else:
    testType = 'trainedClassifier'
    
t_classifier = trainer.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)

# Loop that incrementally adds more and more classes

print(args.step_size)

train_start = 0
train_end = args.base_classes
test_start = 0
test_end = args.base_classes
total_epochs = args.nepochs
schedule = np.array(args.schedule)

tasknum = (dataset.classes-args.base_classes)//args.step_size+1

results = {}
for head in ['all', 'prev_new', 'task', 'cheat']:
    results[head] = {}
    results[head]['correct'] = []
    results[head]['correct_5'] = []
    results[head]['stat'] = []
    
results['task_soft_1'] = np.zeros((tasknum, tasknum))
results['task_soft_5'] = np.zeros((tasknum, tasknum))

print(tasknum)

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
            
        if epoch % 10 == (10 - 1) and args.debug:
            if args.trainer == 'icarl':
                incremental_loader.mode = 'evaluate'
                t_classifier.update_moment(myTrainer.model, train_iterator, args.step_size, t)
            
            if t>0:
                ###################### 폐기처분 대상 ######################
                incremental_loader.mode = 'evaluate'
                train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, 0, train_end)
                print("*********CURRENT EPOCH********** : %d"%epoch)
                print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                print("Train Classifier top-5 (Softmax): %0.2f"%train_5)

                incremental_loader.mode = 'test'
                correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                                 test_start, test_end, 
                                                                 mode='test', step_size=args.step_size)


                print("Test Classifier top-1 (Softmax, all): %0.2f"%correct['all'])
                print("Test Classifier top-5 (Softmax, all): %0.2f"%correct_5['all'])
                print("Test Classifier top-1 (Softmax, prev_new): %0.2f"%correct['prev_new'])
                print("Test Classifier top-5 (Softmax, prev_new): %0.2f"%correct_5['prev_new'])
                
            else:
                ###################### 폐기처분 대상 ######################
                incremental_loader.mode = 'evaluate'
                train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, 0, train_end)
                print("*********CURRENT EPOCH********** : %d"%epoch)
                print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                print("Train Classifier top-5 (Softmax): %0.2f"%train_5)
                
                incremental_loader.mode = 'test'
                test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                          mode='test', step_size=args.step_size)
                print("Test Classifier top-1 (Softmax): %0.2f"%test_1)
                print("Test Classifier top-5 (Softmax): %0.2f"%test_5)
            
    
    # Evaluate the learned classifier
    if args.trainer == 'icarl':
        incremental_loader.mode = 'evaluate'
        t_classifier.update_moment(myTrainer.model, train_iterator, args.step_size, t)
        print('Moment update finished')
    
    if args.trainer == 'il2m':
        incremental_loader.mode = 'evaluate'
        t_classifier.update_mean(myTrainer.model, train_iterator, train_end, args.step_size)
        print('Mean update finished')
    
    
    if t > 0 and (args.trainer == 'ft_wa' or args.trainer == 'wa'):
        myTrainer.weight_align()   
    
    if t > 0 and args.trainer == 'eeil':
        myTrainer.update_frozen_model()
        myTrainer.balance_fine_tune()
    
    ############################################
    #        BIC bias correction train         #
    ############################################
    
    if 'bic' in args.trainer and t>0:
        incremental_loader.mode = 'bias'
        bias_iterator = torch.utils.data.DataLoader(bias_dataset_loader, 
                                                    batch_size=args.batch_size, shuffle=True, **kwargs)
        print(myTrainer.bias_correction_layer.alpha)
        print(myTrainer.bias_correction_layer.beta)
        
        for e in range(total_epochs*2):
            myTrainer.train_bias_correction(bias_iterator)
            myTrainer.update_bias_lr(e, schedule)
            
            print(myTrainer.bias_correction_layer.alpha)
            print(myTrainer.bias_correction_layer.beta)
            
    if t>0:
        ###################### 폐기처분 대상 ######################
        
        if 'bic' in args.trainer:
            incremental_loader.mode = 'evaluate'
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, 
                                                     0, train_end, myTrainer.bias_correction_layer)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
            incremental_loader.mode = 'test'
            correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
            
        else :
            incremental_loader.mode = 'evaluate'
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
            incremental_loader.mode = 'test'
            correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                             test_start, test_end, 
                                                             mode='test', step_size=args.step_size)

        print("Test Classifier top-1 (Softmax, all): %0.2f"%correct['all'])
        print("Test Classifier top-5 (Softmax, all): %0.2f"%correct_5['all'])
        print("Test Classifier top-1 (Softmax, prev_new): %0.2f"%correct['prev_new'])
        print("Test Classifier top-5 (Softmax, prev_new): %0.2f"%correct_5['prev_new'])
        
        for head in ['all', 'prev_new', 'task']:

            results[head]['correct'].append(correct[head])
            results[head]['correct_5'].append(correct_5[head])
            results[head]['stat'].append(stat[head])


        results['cheat']['correct'].append(correct['cheat'])
        results['cheat']['correct_5'].append(correct_5['cheat'])
        
    else:
        ###################### 폐기처분 대상 ######################
        
        if  'bic' in args.trainer:
            
            incremental_loader.mode = 'evaluate'
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, 
                                                     train_start, train_end, myTrainer.bias_correction_layer)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
            incremental_loader.mode = 'test'
            test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
        
        else :
            incremental_loader.mode = 'evaluate'
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
            print("Train Classifier top-1 Final(Softmax): %0.2f"%train_1)
            print("Train Classifier top-5 Final(Softmax): %0.2f"%train_5)
            
            incremental_loader.mode = 'test'
            test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                              mode='test', step_size=args.step_size)
            
        print("Test Classifier top-1 Final(Softmax): %0.2f"%test_1)
        print("Test Classifier top-5 Final(Softmax): %0.2f"%test_5)
        for head in ['all', 'prev_new', 'task', 'cheat']:
            results[head]['correct'].append(test_1)
            results[head]['correct_5'].append(test_5)
    
    start = 0
    end = args.base_classes
    result_loader.reset()
    iterator = torch.utils.data.DataLoader(result_loader, batch_size=100, **kwargs)
    for i in range(t+1):
        
        if 'bic' in args.trainer:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end,
                                                                                              myTrainer.bias_correction_layer)
        else:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end)
        start = end
        end += args.step_size
        
        result_loader.task_change()
    
    sio.savemat('./result_data/'+log_name+'.mat',results)
    
    if args.trainer == 'ssil':
        train_start = train_end - args.step_size
    
    torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))
    if 'bic' in args.trainer:
        torch.save(myTrainer.bias_correction_layer.state_dict(), 
                   './models/trained_model/' + log_name + '_bias' + '_task_{}.pt'.format(t))
    
    myTrainer.increment_classes()
#     evaluate_dataset_loader.update_exemplar()
#     evaluate_dataset_loader.task_change()
#     bias_dataset_loader.update_exemplar()
#     bias_dataset_loader.task_change()
    
    train_end = train_end + args.step_size
    test_end = test_end + args.step_size

    
    
# train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
#                                                       dataset.train_labels,
#                                                       dataset.classes,
#                                                       args.step_size,
#                                                       args.memory_budget,
#                                                       'train',
#                                                       transform=dataset.train_transform,
#                                                       loader=loader,
#                                                       shuffle_idx = shuffle_idx,
#                                                       base_classes = args.base_classes,
#                                                       approach = args.trainer,
#                                                       model = myModel
#                                                       )
# # Loader for evaluation
# evaluate_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
#                                                          dataset.train_labels,
#                                                          dataset.classes,
#                                                          args.step_size,
#                                                          args.memory_budget,
#                                                          'train',
#                                                          transform=dataset.train_transform,
#                                                          loader=loader,
#                                                          shuffle_idx = shuffle_idx,
#                                                          base_classes = args.base_classes,
#                                                          approach = 'ft',
#                                                          model = myModel
#                                                         )

# # Loader for test data.
# test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data,
#                                                      dataset.test_labels,
#                                                      dataset.classes,
#                                                      args.step_size,
#                                                      args.memory_budget,
#                                                      'test',
#                                                      transform=dataset.test_transform,
#                                                      loader=loader,
#                                                      shuffle_idx = shuffle_idx,
#                                                      base_classes = args.base_classes,
#                                                      approach = args.trainer,
#                                                      model = myModel
#                                                      )

# # Loader for training bias correction layer in Large-scale Incremental Learning
# bias_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
#                                                      dataset.train_labels,
#                                                      dataset.classes,
#                                                      args.step_size,
#                                                      args.memory_budget,
#                                                      'bias',
#                                                      transform=dataset.train_transform,
#                                                      loader=loader,
#                                                      shuffle_idx = shuffle_idx,
#                                                      base_classes = args.base_classes,
#                                                      approach = args.trainer,
#                                                      model = myModel
#                                                      )