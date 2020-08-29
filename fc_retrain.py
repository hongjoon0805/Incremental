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

myModel = networks.ModelFactory.get_model(args.dataset)
myModel = torch.nn.DataParallel(myModel).cuda()

train_dataset_loader = data_handler.FullLoader(dataset.train_data,
                                                      dataset.train_labels,
                                                      dataset.classes,
                                                      args.step_size,
                                                      args.memory_budget,
                                                      'train',
                                                      transform=dataset.train_transform,
                                                      loader=loader,
                                                      shuffle_idx = shuffle_idx,
                                                      base_classes = args.base_classes,
                                                      approach = args.trainer,
                                                      model = myModel
                                                      )

# Loader for evaluation
evaluate_dataset_loader = data_handler.FullLoader(dataset.train_data,
                                                         dataset.train_labels,
                                                         dataset.classes,
                                                         args.step_size,
                                                         args.memory_budget,
                                                         'train',
                                                         transform=dataset.train_transform,
                                                         loader=loader,
                                                         shuffle_idx = shuffle_idx,
                                                         base_classes = args.base_classes,
                                                         approach = 'ft',
                                                         model = myModel
                                                        )

# Loader for test data.
test_dataset_loader = data_handler.FullLoader(dataset.test_data,
                                                     dataset.test_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'test',
                                                     transform=dataset.test_transform,
                                                     loader=loader,
                                                     shuffle_idx = shuffle_idx,
                                                     base_classes = args.base_classes,
                                                     approach = args.trainer,
                                                     model = myModel
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

test_iterator = torch.utils.data.DataLoader(test_dataset_loader, batch_size=100, shuffle=False, **kwargs)


evaluator_iterator = torch.utils.data.DataLoader(evaluate_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

# Get the required model
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


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
    if t == 0:
        myTrainer.increment_classes(t+1)
        evaluate_dataset_loader.task_change(t+1)
        test_end = test_end + args.step_size
        train_end = train_end + args.step_size
        continue
    flag = 0
    #model load
    model_name = '200804_FT_BIC_Imagenet_ft_bic_0_memsz_20000_base_100_step_100_batch_128_epoch_25_task_{}'.format(t)
    myTrainer.model.load_state_dict(torch.load('models/trained_model/'+model_name+'.pt'))
    #feature frozen
    # Running nepochs epochs
    print('Flag: %d'%flag)
    myTrainer.update_frozen_model()

    for epoch in range(total_epochs):
        if flag:
            break
        
        # upldate_lr
        myTrainer.update_lr(epoch, schedule)
        # train
        myTrainer.train(epoch)
        
        if epoch % 2 == (1 - 1) and args.debug:
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
        
    if t>0:
        ###################### 폐기처분 대상 ######################
        if flag:
            print('Evaluation!')
        if 'bic' in args.trainer:
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
        
        if  'bic' in args.trainer:
            train_1, train_5 = t_classifier.evaluate(myTrainer.model, evaluator_iterator, 
                                                     train_start, train_end, myTrainer.bias_correction_layer)
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
        
        if 'bic' in args.trainer:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end,
                                                                                              myTrainer.bias_correction_layer)
        else:
            results['task_soft_1'][t][i], results['task_soft_5'][t][i] = t_classifier.evaluate(myTrainer.model, 
                                                                                               iterator, start, end)
        start = end
        end += args.step_size


    sio.savemat('./result_data/'+model_name+'_FC'+'.mat',results)
    
    torch.save(myModel.state_dict(), './models/trained_model/' + model_name + '_FC.pt')
    
    if t < tasknum-1:
        myTrainer.increment_classes(t+1)
        evaluate_dataset_loader.task_change(t+1)

    test_end = test_end + args.step_size
    train_end = train_end + args.step_size
