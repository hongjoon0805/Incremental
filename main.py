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

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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


if args.trainer == 'ssil':
    log_name += '_replay_{}'.format(args.replay_batch_size)

if args.trainer == 'ssil' or args.trainer == 'ft' or args.trainer == 'il2m':
    log_name += '_factor_{}'.format(args.factor)
if args.prev_new:
    log_name += '_prev_new'

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.deterministic = True
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
print(shuffle_idx)
train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                      dataset.train_labels,
                                                      dataset.classes,
                                                      args.step_size,
                                                      args.memory_budget,
                                                      'train',
                                                      transform=dataset.train_transform,
                                                      loader=loader,
                                                      shuffle_idx = shuffle_idx,
                                                      base_classes = args.base_classes,
                                                      approach = args.trainer
                                                      )
# Loader for evaluation
evaluate_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                         dataset.train_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         args.memory_budget,
                                                         'train',
                                                         transform=dataset.train_transform,
                                                         loader=loader,
                                                         shuffle_idx = shuffle_idx,
                                                         base_classes = args.base_classes,
                                                         approach = 'ft'
                                                        )

# Loader for test data.
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

# Loader for training bias correction layer in Large-scale Incremental Learning
bias_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                     dataset.train_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'bias',
                                                     transform=dataset.train_transform,
                                                     loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     approach = args.trainer
                                                     )


result_dataset_loaders = data_handler.make_ResultLoaders(dataset.test_data,
                                                         dataset.test_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         transform=dataset.test_transform,
                                                         loader=loader,
                                                         shuffle_idx = shuffle_idx,
                                                         base_classes = args.base_classes
                                                        )

# Iterator to iterate over training data.
kwargs = {'num_workers': args.workers, 'pin_memory': True}
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

evaluator_iterator = torch.utils.data.DataLoader(evaluate_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

# Iterator to iterate over test data
test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=100, shuffle=False, **kwargs)

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()

# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)


# Initilize the evaluators used to measure the performance of the system.

if args.trainer == 'icarl':
    testType = "generativeClassifier"
elif args.trainer == 'il2m':
    testType = 'il2m'
elif args.trainer == 'bic':
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
    
    if args.trainer == 'il2m' :
        model_name = 'models/trained_model/ECCV_final_{}_ft_{}_memsz_{}_base_{}_step_{}_batch_128_epoch_100_factor_4_task_{}.pt'.format(args.dataset, args.seed, args.memory_budget, args.base_classes, args.step_size, t)
        myTrainer.model.load_state_dict(torch.load(model_name))
    
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr
    if args.trainer == 'ssil' or args.trainer == 'ft':
        lr = args.lr / (t+1)
        if t==1:
            total_epochs = args.nepochs // args.factor
            schedule = schedule // args.factor
    
    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)
    flag = 0
    mem_base = {}
    mem_base['Imagenet'] = 5000
    mem_base['Google_Landmark_v2_1K'] = 5000
    mem_base['Google_Landmark_v2_10K'] = 10000
    if args.trainer == 'ft' and t==0:
        try:
            model_name = 'models/trained_model/ECCV_final_{}_ft_{}_memsz_{}_base_{}_step_{}_batch_128_epoch_100_factor_4_task_{}.pt'.format(args.dataset, args.seed, mem_base[args.dataset], args.base_classes, args.step_size, t)
            myTrainer.model.load_state_dict(torch.load(model_name))
            flag=1
        except:
            pass
    if args.trainer == 'ssil' and t==0:
        try:
            model_name = 'models/trained_model/ECCV_final_{}_ssil_{}_memsz_{}_base_{}_step_{}_batch_128_epoch_100_replay_32_factor_5_task_{}.pt'.format(args.dataset, args.seed, mem_base[args.dataset], args.base_classes, args.step_size, t)
            myTrainer.model.load_state_dict(torch.load(model_name))
            flag=1
        except:
            pass
    
    # Running nepochs epochs
    print('Flag: %d'%flag)
    for epoch in range(0, total_epochs):
        if flag:
            break
        myTrainer.update_lr(epoch, schedule)
        if args.trainer == 'il2m':
            break
        else:
            myTrainer.train(epoch)
        if args.trainer == 'wa' and t > 0:
            myTrainer.weight_align()
        if epoch % 50 == (50 - 1) and args.debug:
            if args.trainer == 'icarl':
                t_classifier.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
            
            if t>0:
                ###################### 폐기처분 대상 ######################
                train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
                print("*********CURRENT EPOCH********** : %d"%epoch)
                print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                print("Train Classifier top-5 (Softmax): %0.2f"%train_5)

                correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                                 test_start, test_end, 
                                                                 mode='test', step_size=args.step_size)


                print("Test Classifier top-1 (Softmax, all): %0.2f"%correct['all'])
                print("Test Classifier top-5 (Softmax, all): %0.2f"%correct_5['all'])
                print("Test Classifier top-1 (Softmax, prev_new): %0.2f"%correct['prev_new'])
                print("Test Classifier top-5 (Softmax, prev_new): %0.2f"%correct_5['prev_new'])
                
            else:
                ###################### 폐기처분 대상 ######################
                train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
                print("*********CURRENT EPOCH********** : %d"%epoch)
                print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                print("Train Classifier top-5 (Softmax): %0.2f"%train_5)
                test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                          mode='test', step_size=args.step_size)
                print("Test Classifier top-1 (Softmax): %0.2f"%test_1)
                print("Test Classifier top-5 (Softmax): %0.2f"%test_5)
            
    
    # Evaluate the learned classifier
    if args.trainer == 'icarl':
        t_classifier.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
        print('Moment update finished')
    
    if args.trainer == 'il2m':
        t_classifier.update_mean(myTrainer.model, evaluator_iterator, train_end, args.step_size)
        print('Mean update finished')
    
    
        
    
    ############################################
    #        BIC bias correction train         #
    ############################################
    
    if args.trainer == 'bic' and t>0 and flag != 1:
        
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
        if flag:
            print('Evaluation!')
        if args.trainer == 'bic':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     0, train_end, myTrainer.bias_correction_layer)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
            correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
            
        else :
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
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
        if flag:
            print('Evaluation!')
        
        if args.trainer == 'bic':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     train_start, train_end, myTrainer.bias_correction_layer,)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
            
            test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
        
        else :
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, train_start, train_end)
            print("Train Classifier top-1 Final(Softmax): %0.2f"%train_1)
            print("Train Classifier top-5 Final(Softmax): %0.2f"%train_5)
            
            test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                              mode='test', step_size=args.step_size)
            
        print("Test Classifier top-1 Final(Softmax): %0.2f"%test_1)
        print("Test Classifier top-5 Final(Softmax): %0.2f"%test_5)
        for head in ['all', 'prev_new', 'task', 'cheat']:
            results[head]['correct'].append(test_1)
            results[head]['correct_5'].append(test_5)
    
    start = 0
    end = args.base_classes
    for i in range(t+1):
        dataset_loader = result_dataset_loaders[i]
        iterator = torch.utils.data.DataLoader(dataset_loader,
                                               batch_size=args.batch_size, **kwargs)
        
        if args.trainer == 'bic':
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end,
                                                                                              myTrainer.bias_correction_layer)
        else:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end)
        start = end
        end += args.step_size
    
    sio.savemat('./result_data/'+log_name+'.mat',results)
    
    if args.trainer == 'ssil':
        train_start = train_end - args.step_size
    if args.trainer == 'ssil' or args.trainer == 'ft' or args.trainer == 'icarl':
        torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))
    if args.trainer == 'bic' :
        torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))
        torch.save(myTrainer.bias_correction_layer.state_dict(), 
                   './models/trained_model/' + log_name + '_bias' + '_task_{}.pt'.format(t))
        
    myTrainer.increment_classes()
    evaluate_dataset_loader.update_exemplar()
    evaluate_dataset_loader.task_change()
    bias_dataset_loader.update_exemplar()
    bias_dataset_loader.task_change()
    
    train_end = train_end + args.step_size
    test_end = test_end + args.step_size
