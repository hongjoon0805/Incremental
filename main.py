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

torch.set_default_tensor_type('torch.cuda.FloatTensor')

args = arguments.get_args()

log_name = '{}_{}_{}_{}_memsz_{}_alpha_{}_ratio_{:.8f}_beta_{}_lr_{}_out_{}_batch_{}_epoch_{}'.format(args.date,
                                                                                                      args.dataset,
                                                                                                      args.trainer,
                                                                                                      args.seed,
                                                                                                      args.memory_budget,
                                                                                                      args.alpha,
                                                                                                      args.ratio,
                                                                                                      args.beta,
                                                                                                      args.lr,
                                                                                                      args.out_channels,
                                                                                                      args.batch_size,
                                                                                                      args.epochs_class)


dataset = data_handler.DatasetFactory.get_dataset(args.dataset)
seed = args.seed
m = args.memory_budget

# Fix the seed.
args.seed = seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Loader used for training data
train_dataset_loader = data_handler.IncrementalLoader(dataset.train_data.train_data,
                                                      dataset.train_data.train_labels,
                                                      dataset.classes,
                                                      args.step_size,
                                                      args.memory_budget,
                                                      'train',
                                                      args.batch_size,
                                                      transform=dataset.train_transform,
                                                      )

# Loader for test data.
test_dataset_loader = data_handler.IncrementalLoader(dataset.test_data.test_data,
                                                     dataset.test_data.test_labels,
                                                     dataset.classes,
                                                     args.step_size,
                                                     args.memory_budget,
                                                     'test',
                                                     args.batch_size,
                                                     transform=dataset.test_transform
                                                     )

kwargs = {'num_workers': 1, 'pin_memory': True} 

# Iterator to iterate over training data.
train_iterator = torch.utils.data.DataLoader(train_dataset_loader,
                                             batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

# Iterator to iterate over test data
test_iterator = torch.utils.data.DataLoader(test_dataset_loader,
                                            batch_size=args.batch_size, shuffle=False, **kwargs)

# Get the required model
myModel = networks.ModelFactory.get_model(args.dataset, args.ratio, args.trainer, args.out_channels).cuda()
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
results_soft = np.zeros(dataset.classes // args.step_size)
if args.trainer == 'gda':
    gda_classifier = trainer.EvaluatorFactory.get_evaluator("generativeClassifier")
    results_gda = np.zeros(dataset.classes // args.step_size)


# Loop that incrementally adds more and more classes
for t in range(dataset.classes//args.step_size):
    print("SEED:", seed, "MEMORY_BUDGET:", m, "tasknum:", t)
    # Add new classes to the train, and test iterator
    myTrainer.update_frozen_model()
    epoch = 0

    # Running epochs_class epochs
    for epoch in range(0, args.epochs_class):
        myTrainer.update_lr(epoch)
        myTrainer.train(epoch)
        # print(my_trainer.threshold)
        if epoch % 5 == (5 - 1):
            TrainError_softmax = t_classifier.evaluate(myTrainer.model, train_iterator, t, args.step_size, 'train')
            TestError_softmax = t_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
            print("*********CURRENT EPOCH********** : %d"%epoch)
            print("Train Classifier (Softmax): %0.2f"%TrainError_softmax)
            print("Test Classifier (Softmax): %0.2f"%TestError_softmax)
            if args.trainer == 'gda':
                
                gda_classifier.update_moment(myTrainer.model, train_iterator, args.step_size)
                TrainError_gda = gda_classifier.evaluate(myTrainer.model, train_iterator, t, args.step_size, 'train')
                TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
                
                print("Train Classifier (GDA): %0.2f"%TrainError_gda)
                print("Test Classifier (GDA): %0.2f"%TestError_gda)
            

    
    # Evaluate the learned classifier
    
    # t-SNE visualization tool 짜놓기
    # 각 task에 대한 test accuracy를 만들 수 있는 dataloader & output 코드 짜놓기.
    # class incremental accuracy, task test accuracy(use all heads), task test accuracy(use step-size head)
    # Resovior sampling 자체가 잘못되었나? resorvior sampling이 각 class를 uniform하게 가지고 있는지 알아보자.
    
    
    TestError_softmax = t_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
    print("Test Classifier Final(Softmax): %0.2f"%TestError_softmax)
    results_soft[t] = TestError_softmax
    np.savetxt('./result_data/'+log_name+'_Soft.txt', results_soft, '%.4f')
    
    break
    
    if args.trainer == 'gda':
        gda_classifier.update_moments(myTrainer.model, train_iterator, args.step_size)
        TestError_gda = gda_classifier.evaluate(myTrainer.model, test_iterator, t, args.step_size, 'test')
        print("Test Classifier Final(GDA): %0.2f"%TestError_gda)
        results_gda[t] = TestError_gda
        np.savetxt('./result_data/'+log_name+'_GDA.txt', results_gda, '%.4f')
    
    myTrainer.setup_training()
    torch.save(myModel.state_dict(), './models/trained_model/' + log_name + '_task_{}.pt'.format(t))


