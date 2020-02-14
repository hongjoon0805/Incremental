import argparse

import torch
torch.backends.cudnn.benchmark=True
import torch.utils.data as td
import numpy as np
import scipy.io as sio

import data_handler
import experiment as ex
import networks
import trainer
import arguments
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

torch.set_default_tensor_type('torch.cuda.FloatTensor')

args = arguments.get_args()

if args.trainer == 'bin_finetune':
    log_name = '{}_{}_{}_{}_{}_memsz_{}_base_{}_replay_{}_batch_{}_epoch_{}_factor_{}'.format(
        args.date,
        args.dataset,
        args.trainer,
        args.option,
        args.seed,
        args.memory_budget,
        args.base_classes,
        args.replay_batch_size,
        args.batch_size,
        args.epochs_class,
        args.factor
    )

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
    args.epochs_class,
    args.factor,
    args.strategy,
    args.loss
)

if args.prev_new:
    log_name += '_prev_new'
if args.uniform_penalty:
    log_name += '_uniform_penalty'
if args.CI:
    log_name += '_CI'
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

dataset = data_handler.DatasetFactory.get_dataset(args.dataset)

if args.dataset == 'CIFAR100':
    loader = None
    
elif args.dataset == 'Imagenet':
    loader = dataset.loader
    
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
shuffle_idx = shuffle(np.arange(dataset.classes), random_state=args.seed)
# shuffle_idx = shuffle(np.arange(dataset.classes))
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

kwargs = {'num_workers': 32, 'pin_memory': True}

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
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

# Iterator to iterate over test data
test_iterator = torch.utils.data.DataLoader(test_dataset_loader,
                                            batch_size=100, shuffle=False, **kwargs)

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

myModel = networks.ModelFactory.get_model(args.dataset, args.ratio, args.trainer)
myModel = torch.nn.DataParallel(myModel).cuda()

# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)


# Initilize the evaluators used to measure the performance of the system.
if args.trainer == 'bin_finetune':
    testType = "binaryClassifier"
elif args.trainer == 'er' or args.trainer == 'coreset':
    testType = "trainedClassifier"

t_classifier = trainer.EvaluatorFactory.get_evaluator(testType)
if args.trainer == 'gda':
    gda_classifier = trainer.EvaluatorFactory.get_evaluator("generativeClassifier")
    results_gda = np.zeros(dataset.classes // args.step_size)


# Loop that incrementally adds more and more classes

train_start = 0
train_end = args.base_classes
test_start = 0
test_end = args.base_classes
total_epochs = args.epochs_class
schedule = np.array(args.schedule)

tasknum = (dataset.classes-args.base_classes)//args.step_size+1

results = {}
for head in ['all', 'prev_new', 'task', 'cheat']:
    results[head] = {}
    results[head]['correct'] = []
    results[head]['correct_5'] = []
    results[head]['stat'] = []
    
results['bin_target'] = []
results['bin_prob'] = []
results['sigmoid'] = []
results['auroc'] = []
results['task_soft_1'] = np.zeros((tasknum, tasknum))
results['task_soft_5'] = np.zeros((tasknum, tasknum))

for t in range((dataset.classes-args.base_classes)//args.step_size+1):
    if t==0 and args.trainer == 'bin_finetune':
        myTrainer.increment_classes()
        train_end = train_end + args.step_size
        test_end = test_end + args.step_size
        continue
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    print(len(train_dataset_loader.exemplar))
    # Add new classes to the train, and test iterator
    lr = args.lr
    if args.lr_change:
        lr = args.lr / (t+1)
    
    if t==1:
        total_epochs = args.epochs_class // args.factor
        schedule = schedule // args.factor
#     if t==4:
#         break
    
    if args.trainer == 'ood' and args.rand_init:
        myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)
    myTrainer.update_frozen_model()
    myTrainer.setup_training(lr)
    
    # Running epochs_class epochs
    for epoch in range(0, total_epochs):
        myTrainer.update_lr(epoch, schedule)
        myTrainer.train(epoch)
        # print(my_trainer.threshold)
        if epoch % 5 == (5 - 1):
            
            if t>0:
                if args.trainer == 'bin_finetune':
                    train_dataset_loader.approach = 'coreset'
                    train_dataset_loader.len += args.memory_budget
                    bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
                    train_dataset_loader.approach = args.trainer
                    train_dataset_loader.len -= args.memory_budget
                    auroc = roc_auc_score(bin_target, bin_prob)
                    print("Train Classifier (AUROC): %0.2f"%auroc)

                    bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end)
                    auroc = roc_auc_score(bin_target, bin_prob)
                    print("Test Classifier (AUROC): %0.2f"%auroc)
                elif args.trainer == 'er' or args.trainer == 'coreset':    
                    train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
                    print("*********CURRENT EPOCH********** : %d"%epoch)
                    print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                    print("Train Classifier top-5 (Softmax): %0.2f"%train_5)

                    correct, correct_5, stat, bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                                                           test_start, test_end, 
                                                                                           mode='test', step_size=args.step_size)

                    auroc = roc_auc_score(bin_target, bin_prob)

                    print("Test Classifier top-1 (Softmax, all): %0.2f"%correct['all'])
                    print("Test Classifier top-5 (Softmax, all): %0.2f"%correct_5['all'])
                    print("Test Classifier top-1 (Softmax, prev_new): %0.2f"%correct['prev_new'])
                    print("Test Classifier top-5 (Softmax, prev_new): %0.2f"%correct_5['prev_new'])
                    print("Test Classifier top-1 (Softmax, task): %0.2f"%correct['task'])
                    print("Test Classifier top-5 (Softmax, task): %0.2f"%correct_5['task'])
                    print("Test Classifier top-1 (Softmax, cheat): %0.2f"%correct['cheat'])
                    print("Test Classifier top-5 (Softmax, cheat): %0.2f"%correct_5['cheat'])
                    print("Test Classifier (Binary Classification): %0.2f"%correct['bin'])
                    print("Test Classifier (AUROC): %0.2f"%auroc)
                    for head in ['all', 'prev_new', 'task']:
                        print('Test stat for %s'%head)
                        print('cp: %d'%stat[head][0])
                        print('epp: %d'%stat[head][1])
                        print('epn: %d'%stat[head][2])
                        print('cn: %d'%stat[head][3])
                        print('enn: %d'%stat[head][4])
                        print('enp: %d'%stat[head][5])
                        print('total: %d'%stat[head][6])
                else:
                    train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
                    print("*********CURRENT EPOCH********** : %d"%epoch)
                    print("Train Classifier top-1 (Softmax): %0.2f"%train_1)
                    print("Train Classifier top-5 (Softmax): %0.2f"%train_5)
                    test_1, test_5 = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                              mode='test', step_size=args.step_size)
                    print("Test Classifier top-1 (Softmax): %0.2f"%test_1)
                    print("Test Classifier top-5 (Softmax): %0.2f"%test_5)
            

    
    # Evaluate the learned classifier
    
    # t-SNE visualization tool 짜놓기
    # CutMix로 data 저장하는 
    
    
    
    if t>0:
        if args.trainer == 'bin_finetune':
            train_dataset_loader.approach = 'coreset'
            train_dataset_loader.len += args.memory_budget
            bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, train_iterator, 
                                                         train_start, train_end, step_size=args.step_size)
            train_dataset_loader.approach = args.trainer
            train_dataset_loader.len -= args.memory_budget
            auroc = roc_auc_score(bin_target, bin_prob)
            print("Train Classifier (AUROC): %0.2f"%auroc)

            bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, test_iterator, 
                                                         test_start, test_end, step_size=args.step_size)
            auroc = roc_auc_score(bin_target, bin_prob)
            print("Test Classifier (AUROC): %0.2f"%auroc)
            
            results['bin_target'].append(bin_target)
            results['bin_prob'].append(bin_prob)
            
        elif args.trainer == 'er' or args.trainer == 'coreset':    
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator,train_start, train_end)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier Final top-1 (Softmax): %0.2f"%train_1)
            print("Train Classifier Final top-5 (Softmax): %0.2f"%train_5)

            correct, correct_5, stat, bin_target, bin_prob = t_classifier.evaluate(myTrainer.model, test_iterator,
                                                                                           test_start, test_end, 
                                                                                           mode='test', step_size=args.step_size)

            auroc = roc_auc_score(bin_target, bin_prob)

            print("Test Classifier top-1 (Softmax, all): %0.2f"%correct['all'])
            print("Test Classifier top-5 (Softmax, all): %0.2f"%correct_5['all'])
            print("Test Classifier top-1 (Softmax, prev_new): %0.2f"%correct['prev_new'])
            print("Test Classifier top-5 (Softmax, prev_new): %0.2f"%correct_5['prev_new'])
            print("Test Classifier top-1 (Softmax, task): %0.2f"%correct['task'])
            print("Test Classifier top-5 (Softmax, task): %0.2f"%correct_5['task'])
            print("Test Classifier top-1 (Softmax, cheat): %0.2f"%correct['cheat'])
            print("Test Classifier top-5 (Softmax, cheat): %0.2f"%correct_5['cheat'])
            print("Test Classifier Final(Binary Classification): %0.2f"%correct['bin'])
            print("Test Classifier Final(AUROC): %0.2f"%auroc)
            for head in ['all', 'prev_new', 'task']:
                print('Test stat Final for %s'%head)
                print('cp: %d'%stat[head][0])
                print('epp: %d'%stat[head][1])
                print('epn: %d'%stat[head][2])
                print('cn: %d'%stat[head][3])
                print('enn: %d'%stat[head][4])
                print('enp: %d'%stat[head][5])
                print('total: %d'%stat[head][6])

                results[head]['correct'].append(correct[head])
                results[head]['correct_5'].append(correct_5[head])
                results[head]['stat'].append(stat[head])


            results['cheat']['correct'].append(correct['cheat'])
            results['cheat']['correct_5'].append(correct_5['cheat'])
            results['sigmoid'].append(correct['bin'])
            results['auroc'].append(auroc)
            results['bin_target'].append(bin_target)
            results['bin_prob'].append(bin_prob)
        
    else:
        train_1, train_5 = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
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
    if args.trainer != 'bin_finetune':
        for i in range(t+1):
            dataset_loader = result_dataset_loaders[i]
            iterator = torch.utils.data.DataLoader(dataset_loader,
                                                   batch_size=args.batch_size, **kwargs)
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, iterator, start, end)
            start = end
            end += args.step_size
    
    sio.savemat('./result_data/'+log_name+'.mat',results)
    
    if args.trainer == 'gda':
        gda_classifier.update_moments(myTrainer.model, train_iterator, args.step_size)
        TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
        print("Test Classifier Final(GDA): %0.2f"%TestError_gda)
        results_gda[t] = TestError_gda
        np.savetxt('./result_data/'+log_name+'_GDA.txt', results_gda, '%.4f')
    
    myTrainer.increment_classes()
    train_end = train_end + args.step_size
    test_end = test_end + args.step_size
    if args.trainer == 'er':
        train_start = train_end - args.step_size
    torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))


#             if args.trainer == 'gda':

#                 gda_classifier.update_moment(myTrainer.model, train_iterator, args.step_size)
#                 TrainError_gda = gda_classifier.evaluate(myTrainer.model, train_iterator, t, args.step_size, 'train')
#                 TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')

#                 print("Train Classifier (GDA): %0.2f"%TrainError_gda)
#                 print("Test Classifier (GDA): %0.2f"%TestError_gda)