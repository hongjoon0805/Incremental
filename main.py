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

log_name = '{}_{}_{}_{}_memsz_{}_alpha_{}_beta_{}_base_{}_replay_{}_batch_{}_epoch_{}_factor_{}_{}_{}'.format(
    args.date,
    args.dataset,
    args.trainer,
    args.seed,
    args.memory_budget,
    args.alpha,
    args.beta,
    args.base_classes,
    args.replay_batch_size,
    args.batch_size,
    args.nepochs,
    args.factor,
    args.strategy,
    args.loss
)

if args.trainer == 'er' and args.ablation is not 'None':
    log_name += '_' + args.ablation
if args.prev_new:
    log_name += '_prev_new'
if args.uniform_penalty:
    log_name += '_uniform_penalty'
if args.CI:
    log_name += '_CI'
if args.KD:
    log_name += '_KD'
if args.cutmix:
    log_name += '_CutMix'
if args.alpha<1:
    log_name += '_LabelSmoothing'
if args.rand_init:
    log_name += '_rand_init'
if args.lr_change:
    log_name += '_lr_change'
if args.bin_sigmoid:
    log_name += '_bin_sigmoid'
if args.bin_softmax:
    log_name += '_bin_softmax'

if args.benchmark:
    torch.backends.cudnn.benchmark=True
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
                                                      args.batch_size,
                                                      transform=dataset.train_transform,
                                                      loader=loader,
                                                      shuffle_idx = shuffle_idx,
                                                      base_classes = args.base_classes,
                                                      strategy = args.strategy,
                                                      approach = args.trainer
                                                      )
# Loader for evaluation
evaluate_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                         dataset.train_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         args.memory_budget,
                                                         'train',
                                                         args.batch_size,
                                                         transform=dataset.train_transform,
                                                         loader=loader,
                                                         shuffle_idx = shuffle_idx,
                                                         base_classes = args.base_classes,
                                                         strategy = args.strategy,
                                                         approach = 'coreset'
                                                        )

# Loader for test data.
test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data,
                                                     dataset.test_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'test',
                                                     args.batch_size,
                                                     transform=dataset.test_transform,
                                                     loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     strategy = args.strategy,
                                                     approach = args.trainer
                                                     )

# Loader for training bias correction layer in Large-scale Incremental Learning
bias_dataset_loader = data_handler.IncrementalLoader(dataset.train_data,
                                                     dataset.train_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'bias',
                                                     args.batch_size,
                                                     transform=dataset.train_transform,
                                                     loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     strategy = args.strategy,
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

if args.trainer == 'coreset_NMC' or args.trainer == 'er_NMC' or args.trainer == 'icarl':
    testType = "generativeClassifier"
elif args.trainer == 'IL2M':
    testType = 'IL2M'
elif args.trainer == 'bic':
    testType = 'bic'
else:
    testType = 'trainedClassifier'
    
t_classifier = trainer.EvaluatorFactory.get_evaluator(testType, classes=dataset.classes)

# Loop that incrementally adds more and more classes

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

for t in range((dataset.classes-args.base_classes)//args.step_size+1):
    
    if args.trainer == 'IL2M' or args.trainer == 'coreset_NMC':
        model_name = 'models/trained_model/RESULT_{}_coreset_{}_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_128_epoch_100_factor_4_RingBuffer_CE_lr_change_task_{}.pt'.format(args.dataset, args.seed, args.memory_budget, args.base_classes, t)
        myTrainer.model.load_state_dict(torch.load(model_name))
        
    elif args.trainer == 'er_NMC':
        model_name = 'models/trained_model/RESULT_{}_er_{}_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_128_epoch_100_factor_5_RingBuffer_CE_lr_change_task_{}.pt'.format(args.dataset, args.seed, args.memory_budget, args.base_classes, t)
        myTrainer.model.load_state_dict(torch.load(model_name))
    
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr
    if args.lr_change:
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
    if (args.trainer == 'er' or args.trainer == 'coreset') and t==0:
        try:
            model_name = 'models/trained_model/RESULT_{}_coreset_{}_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_128_epoch_100_factor_4_RingBuffer_CE_lr_change_task_{}.pt'.format(args.dataset, args.seed, mem_base[args.dataset], args.base_classes, t)
            myTrainer.model.load_state_dict(torch.load(model_name))
            flag=1
        except:
            pass
        
    if args.trainer == 'icarl':
        try:
            model_name = 'models/trained_model/RESULT_{}_icarl_{}_memsz_20000_alpha_1_beta_0.0001_base_{}_replay_32_batch_128_epoch_60_factor_1_RingBuffer_CE_task_{}.pt'.format(args.dataset, args.seed, args.base_classes, t)
            myTrainer.model.load_state_dict(torch.load(model_name))
            flag=1
        except:
            pass
        
    if args.trainer == 'bic':
        try:
            model_name = 'models/trained_model/ECCV_rebuttal_{}_bic_0_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_256_epoch_100_factor_1_RingBuffer_CE_task_{}.pt'.format(args.dataset, args.memory_budget, args.base_classes, t)
            myTrainer.model.load_state_dict(torch.load(model_name))
            
            model_name = 'models/trained_model/ECCV_rebuttal_{}_bic_0_memsz_{}_alpha_1_beta_0.0001_base_{}_replay_32_batch_256_epoch_100_factor_1_RingBuffer_CE_bias_task_{}.pt'.format(args.dataset, args.memory_budget, args.base_classes, t)
            myTrainer.bias_correction_layer.load_state_dict(torch.load(model_name))
            flag=1
        except:
            pass
    
    # Running nepochs epochs
    print('Flag: %d'%flag)
    for epoch in range(0, total_epochs):
        if flag:
            break
        myTrainer.update_lr(epoch, schedule)
        if args.trainer == 'IL2M' or args.trainer == 'coreset_NMC' or args.trainer == 'er_NMC':
            break
        else:
            myTrainer.train(epoch)
        
        if epoch % 5 == (5 - 1) and args.debug:
            if args.trainer == 'icarl':
                t_classifier.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
            
            if t>0:
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
                train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
                print("*********CURRENT EPOCH********** : %d"%epoch)
                print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                print("Train Classifier top-5 (Softmax): %0.2f"%train_5)
                test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                          mode='test', step_size=args.step_size)
                print("Test Classifier top-1 (Softmax): %0.2f"%test_1)
                print("Test Classifier top-5 (Softmax): %0.2f"%test_5)
            
    
    # Evaluate the learned classifier
    if args.trainer == 'icarl' or args.trainer == 'coreset_NMC' or args.trainer == 'er_NMC':
        t_classifier.update_moment(myTrainer.model, evaluator_iterator, args.step_size, t)
        print('Moment update finished')
    
    if args.trainer == 'IL2M':
        t_classifier.update_mean(myTrainer.model, evaluator_iterator, train_end, args.step_size)
        print('Mean update finished')
    
    ############################################
    #        BIC bias correction train         #
    ############################################
    
    if args.trainer == 'bic' and t>0 and flag != 1:
        
        train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     0, train_end, myTrainer.bias_correction_layer)
        print("*********CURRENT EPOCH********** : %d"%epoch)
        print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
        print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
        
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
        
        if flag:
            print('Evaluation!')
        
        if args.trainer == 'er' or args.trainer == 'coreset' or args.trainer == 'icarl':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 0, train_end)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
        
        elif args.trainer == 'bic':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     0, train_end, myTrainer.bias_correction_layer)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)

        if args.trainer == 'bic':
            correct, correct_5, stat = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
        else:
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
        if flag:
            print('Evaluation!')
        
        if args.trainer == 'er' or args.trainer == 'coreset' or args.trainer == 'icarl':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, train_start, train_end)
            print("Train Classifier top-1 Final(Softmax): %0.2f"%train_1)
            print("Train Classifier top-5 Final(Softmax): %0.2f"%train_5)
        elif args.trainer == 'bic':
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     train_start, train_end, myTrainer.bias_correction_layer,)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)
        
        if args.trainer == 'bic':
            test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                         test_start, test_end, myTrainer.bias_correction_layer,
                                                         mode='test', step_size=args.step_size)
        
        else:
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
    
    myTrainer.increment_classes()
    evaluate_dataset_loader.update_exemplar()
    evaluate_dataset_loader.task_change()
    bias_dataset_loader.update_exemplar()
    bias_dataset_loader.task_change()
    
    train_end = train_end + args.step_size
    test_end = test_end + args.step_size
    if args.trainer == 'er':
        train_start = train_end - args.step_size
    if args.trainer == 'er' or args.trainer == 'coreset' or args.trainer == 'icarl':
        torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))
    if args.trainer == 'bic' :
        torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))
        torch.save(myTrainer.bias_correction_layer.state_dict(), 
                   './models/trained_model/' + log_name + '_bias' + '_task_{}.pt'.format(t))
