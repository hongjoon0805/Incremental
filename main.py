import argparse

import torch
torch.backends.cudnn.benchmark=True
import torch.utils.data as td
import numpy as np

import data_handler
import experiment as ex
import networks
import trainer
import arguments
from sklearn.utils import shuffle

torch.set_default_tensor_type('torch.cuda.FloatTensor')

args = arguments.get_args()

log_name = '{}_{}_{}_{}_memsz_{}_alpha_{}_base_{}_batch_{}_epoch_{}_factor_{}_{}'.format(
    args.date,
    args.dataset,
    args.trainer,
    args.seed,
    args.memory_budget,
    args.alpha,
    args.base_classes,
    args.batch_size,
    args.epochs_class,
    args.factor,
    args.strategy
)


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

kwargs = {'num_workers': 16, 'pin_memory': True}

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
myModel = networks.ModelFactory.get_model(args.dataset, args.ratio, args.trainer)
myModel = torch.nn.DataParallel(myModel).cuda()
# myModel = networks.ModelFactory.get_model(args.dataset).cuda()

# Define the optimizer used in the experiment
optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)
# if args.trainer == 'bayes':
#     optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum, nesterov=True)

# Trainer object used for training
myTrainer = trainer.TrainerFactory.get_trainer(train_iterator, test_iterator, dataset, myModel, args, optimizer)

# Initilize the evaluators used to measure the performance of the system.
t_classifier = trainer.EvaluatorFactory.get_evaluator("trainedClassifier")
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

results_soft = []
results_stat = []
results_task_soft = np.zeros((tasknum, tasknum))
stat_per_class = np.zeros(dataset.classes)
stat_confidence = np.zeros(tasknum)

for t in range((dataset.classes-args.base_classes)//args.step_size+1):
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    lr = args.lr / (t+1)
    
    if t==1:
        total_epochs = args.epochs_class // args.factor
        schedule = schedule // args.factor

    myTrainer.update_frozen_model()
    # Running epochs_class epochs
    for epoch in range(0, total_epochs):
        myTrainer.update_lr(epoch, schedule)
        myTrainer.train(epoch)
        # print(my_trainer.threshold)
        if epoch % 5 == (5 - 1):
            
            TrainError_softmax = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier (Softmax): %0.2f"%TrainError_softmax)
            if t>0:
                TestError_softmax, stat_list = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                          mode='test', step_size=args.step_size)
                print("Test Classifier (Softmax): %0.2f"%TestError_softmax)
                print('cp: %d'%stat_list[0])
                print('epp: %d'%stat_list[1])
                print('epn: %d'%stat_list[2])
                print('cn: %d'%stat_list[3])
                print('enn: %d'%stat_list[4])
                print('enp: %d'%stat_list[5])
                print('total: %d'%stat_list[6])
            else:
                TestError_softmax = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                                          mode='test', step_size=args.step_size)
                print("Test Classifier (Softmax): %0.2f"%TestError_softmax)
            
            if args.trainer == 'gda':
                
                gda_classifier.update_moment(myTrainer.model, train_iterator, args.step_size)
                TrainError_gda = gda_classifier.evaluate(myTrainer.model, train_iterator, t, args.step_size, 'train')
                TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
                
                print("Train Classifier (GDA): %0.2f"%TrainError_gda)
                print("Test Classifier (GDA): %0.2f"%TestError_gda)
    
    # Evaluate the learned classifier
    
    # t-SNE visualization tool 짜놓기
    # prev --> new, new --> prev, prev --> prev 로 가는 error cound 결과 짜놓기.
    
    TrainError_softmax = t_classifier.evaluate(myTrainer.model, train_iterator, train_start, train_end)
    print("Train Classifier Final(Softmax): %0.2f"%TrainError_softmax)
    
    if t>0:
        TestError_softmax, stat_list = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                              mode='test', step_size=args.step_size)
        print("Test Classifier Final(Softmax): %0.2f"%TestError_softmax)
        print('cp: %d'%stat_list[0])
        print('epp: %d'%stat_list[1])
        print('epn: %d'%stat_list[2])
        print('cn: %d'%stat_list[3])
        print('enn: %d'%stat_list[4])
        print('enp: %d'%stat_list[5])
        print('total: %d'%stat_list[6])
    
        results_soft.append(TestError_softmax)
        results_stat.append(stat_list)
        np.savetxt('./result_data/'+log_name+'_Soft.txt', np.array(results_soft), '%.4f')
        np.savetxt('./result_data/'+log_name+'_Stat.txt', np.array(results_stat), '%d')
    else:
        TestError_softmax = t_classifier.evaluate(myTrainer.model, test_iterator, test_start, test_end, 
                                              mode='test', step_size=args.step_size)
        print("Test Classifier Final(Softmax): %0.2f"%TestError_softmax)
        results_soft.append(TestError_softmax)
        np.savetxt('./result_data/'+log_name+'_Soft.txt', np.array(results_soft), '%.4f')
    
    start = 0
    end = args.base_classes
    for i in range(t+1):
        dataset_loader = result_dataset_loaders[i]
        iterator = torch.utils.data.DataLoader(dataset_loader,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
        results_task_soft[t][i] = t_classifier.evaluate(myTrainer.model, iterator, start, end)
        start = end
        end += args.step_size
    
    np.savetxt('./result_data/'+log_name+'_Soft_all.txt', results_task_soft, '%.4f')
    
    if args.trainer == 'gda':
        gda_classifier.update_moments(myTrainer.model, train_iterator, args.step_size)
        TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
        print("Test Classifier Final(GDA): %0.2f"%TestError_gda)
        results_gda[t] = TestError_gda
        np.savetxt('./result_data/'+log_name+'_GDA.txt', results_gda, '%.4f')
    
    myTrainer.setup_training(lr)
    train_end = train_end + args.step_size
    test_end = test_end + args.step_size
    if args.trainer == 'er':
        train_start = train_end - args.step_size
    torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))


